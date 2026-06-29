# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import time
from pathlib import Path
from typing import Optional

from olive.data.config import DataConfig
from olive.hardware import AcceleratorSpec
from olive.hardware.accelerator import Device
from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


def _infer_shape(dynamic_shape, known_values=None):
    default_values = {
        "batch_size": 1,
        "past_sequence_length": 2,
        "total_sequence_length": 3,
        "sequence_length": 1,
    }
    if known_values:
        default_values.update(known_values)
    inferred_shape = []
    for dim in dynamic_shape:
        if isinstance(dim, int):
            inferred_shape.append(dim)
            continue
        if dim not in default_values:
            raise KeyError(
                f"Unsupported symbolic dimension '{dim}' in shape {dynamic_shape}. "
                f"Known symbols are: {sorted(default_values)}. "
                "Update OnnxDiscrepancyCheck to handle this new case."
            )
        inferred_shape.append(default_values[dim])
    return tuple(inferred_shape)


def _infer_onnx_weight_dtype(onnx_model):
    """Infer the dominant floating-point dtype used by the ONNX model weights.

    Inspects the model initializers (weights) and returns the most common
    floating-point ONNX TensorProto data type. Returns ``None`` when no
    floating-point initializer is found.
    """
    from collections import Counter

    import onnx

    float_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.DOUBLE,
    }
    counts = Counter()
    for initializer in onnx_model.graph.initializer:
        if initializer.data_type in float_types:
            numel = 1
            for d in initializer.dims:
                numel *= d
            counts[initializer.data_type] += numel
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _onnx_dtype_to_torch(onnx_dtype):
    """Map an ONNX TensorProto floating-point data type to a torch dtype."""
    import onnx
    import torch

    mapping = {
        onnx.TensorProto.FLOAT: torch.float32,
        onnx.TensorProto.FLOAT16: torch.float16,
        onnx.TensorProto.BFLOAT16: torch.bfloat16,
        onnx.TensorProto.DOUBLE: torch.float64,
    }
    return mapping.get(onnx_dtype)


def _onnx_output_to_torch(onnx_output, reference_dtype):
    import torch

    onnx_tensor = torch.as_tensor(onnx_output)
    # ORT may return BFLOAT16 as uint16 because numpy has no bf16; reinterpret whenever we're
    # comparing against a non-integer reference.
    if onnx_tensor.dtype == torch.uint16 and reference_dtype != torch.uint16:
        onnx_tensor = onnx_tensor.view(torch.bfloat16)
    return onnx_tensor


def _longest_common_token_sequence(seq_a: list[int], seq_b: list[int]) -> int:
    """Compute the length of the longest common token sequence starting from the beginning.

    Counts how many tokens match consecutively from the start of both sequences
    before the first divergence.
    """
    length = 0
    for a, b in zip(seq_a, seq_b):
        if a != b:
            break
        length += 1
    return length


def _format_seconds(value: Optional[float]) -> str:
    """Format an optional latency value (in seconds) for logging."""
    return "n/a" if value is None else f"{value:.4f}s"


class OnnxDiscrepancyCheck(Pass):
    """Validates ONNX model outputs against a reference PyTorch model.

    This pass does not transform the model. It runs inference on both the
    ONNX model and a reference PyTorch/HuggingFace model with the same inputs,
    then compares outputs element-wise. It reports:
    - Maximum absolute error (MaxAE)
    - Number of elements where the absolute difference exceeds 0.1
    - Number of elements where the absolute difference exceeds 0.01
    - Inference speedup of ONNX over PyTorch on the target device (or CPU fallback)
    - Longest common token sequence from the beginning between transformers
      generate and ONNX Runtime GenAI generate (when enabled)
    - Time-to-first-token and time-to-first-N-tokens latencies for both transformers
      and ONNX Runtime GenAI generation (when enabled)

    The pass status is marked as failed if any configured threshold is exceeded.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "reference_model_path": PassConfigParam(
                type_=str,
                required=True,
                description="Path to the reference PyTorch/HuggingFace model to compare against.",
            ),
            "report_output_dir": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description=(
                    "Directory where discrepancy check results and reference model are saved. "
                    "If not specified, results are written to the pass cache directory."
                ),
            ),
            "save_reference_model_state_dict": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Save the reference PyTorch model weights (state_dict) alongside the results. "
                    "This allows direct comparison between the reference and optimized models."
                ),
            ),
            "max_mae": PassConfigParam(
                type_=Optional[float],
                default_value=None,
                description=(
                    "Maximum acceptable absolute error. "
                    "If the max absolute difference exceeds this value, the pass fails."
                ),
            ),
            "max_elements_above_0_1": PassConfigParam(
                type_=Optional[int],
                default_value=None,
                description=(
                    "Maximum acceptable number of elements with absolute difference > 0.1. If exceeded, the pass fails."
                ),
            ),
            "max_elements_above_0_01": PassConfigParam(
                type_=Optional[int],
                default_value=None,
                description=(
                    "Maximum acceptable number of elements with absolute difference > 0.01. "
                    "If exceeded, the pass fails."
                ),
            ),
            "genai_model_path": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description=(
                    "Path to the ONNX Runtime GenAI model directory. When provided, the pass "
                    "runs token generation using both transformers and ONNX Runtime GenAI, then "
                    "computes the longest common token sequence from the beginning of their outputs."
                ),
            ),
            "warmup_iterations": PassConfigParam(
                type_=int,
                default_value=3,
                description="Number of warmup iterations before timing inference for speedup measurement.",
            ),
            "timing_iterations": PassConfigParam(
                type_=int,
                default_value=5,
                description=(
                    "Number of timed iterations to measure inference speedup (ONNX vs PyTorch). "
                    "Set to 0 to disable speedup measurement."
                ),
            ),
            "generate_prompt": PassConfigParam(
                type_=str,
                default_value="The capital of France is",
                description="Text prompt used for generation comparison between transformers and GenAI.",
            ),
            "generate_max_new_tokens": PassConfigParam(
                type_=int,
                default_value=32,
                description="Maximum number of new tokens to generate for the token sequence comparison.",
            ),
            "time_to_first_n_tokens": PassConfigParam(
                type_=int,
                default_value=5,
                description=(
                    "Number of leading generated tokens used for the time-to-first-N-tokens latency "
                    "measurement reported for both transformers and ONNX Runtime GenAI."
                ),
            ),
            "min_longest_common_tokens": PassConfigParam(
                type_=Optional[int],
                default_value=None,
                description=(
                    "Minimum acceptable length of the longest common token sequence from the "
                    "beginning between transformers and GenAI outputs. If the actual value is "
                    "below this threshold, the pass fails."
                ),
            ),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        import torch

        from olive.common.config_utils import validate_config
        from olive.common.utils import format_data
        from olive.data.template import dummy_data_config_template
        from olive.model.config.io_config import is_io_config_static

        io_config = model.io_config
        if io_config:
            if is_io_config_static(io_config):
                input_shapes = io_config.get("input_shapes")
            else:
                input_shapes = []
                known = {}
                for shape in io_config.get("input_shapes"):
                    new_shape = _infer_shape(shape, known)
                    input_shapes.append(new_shape)
                    known.update(dict(zip(shape, new_shape)))
            data_config = dummy_data_config_template(
                input_shapes, io_config.get("input_names"), io_config.get("input_types")
            )
            data_config = validate_config(data_config, DataConfig)
            data_config.load_dataset_config.params["max_samples"] = 1
        else:
            raise RuntimeError(
                f"Model IO config is missing for {model.model_path}; cannot generate dummy inputs for discrepancy check."
            )
        # Create dataloader
        dc = data_config.to_data_container()
        dataloader = dc.create_dataloader()

        # Load reference PyTorch model
        from transformers import AutoConfig, AutoModelForCausalLM

        ref_cfg = AutoConfig.from_pretrained(config.reference_model_path)
        architectures = getattr(ref_cfg, "architectures", None) or []
        if not any("ForCausalLM" in arch for arch in architectures):
            raise ValueError(
                "OnnxDiscrepancyCheck currently supports only HuggingFace causal language models (ForCausalLM). "
                f"Got architectures={architectures}"
            )

        ref_model = AutoModelForCausalLM.from_pretrained(config.reference_model_path)
        ref_model.eval()

        # Determine the floating-point dtype used by the ONNX model weights and
        # cast the reference PyTorch model to match, so the comparison uses the
        # same numeric precision for the weights on both sides.
        weight_dtype = None
        onnx_weight_dtype = _infer_onnx_weight_dtype(model.load_model())
        if onnx_weight_dtype is not None:
            weight_dtype = _onnx_dtype_to_torch(onnx_weight_dtype)
        # Prepare ONNX session on the target device (fallback to CPU)
        device = self.accelerator_spec.accelerator_type if self.accelerator_spec else None
        execution_provider = self.accelerator_spec.execution_provider if self.accelerator_spec else None
        if device is None:
            device = Device.CPU
        elif not isinstance(device, Device):
            try:
                device = Device(str(device).lower())
            except ValueError:
                logger.warning("Unknown accelerator_type=%s; falling back to CPU.", device)
                device = Device.CPU

        # Determine the torch device matching the accelerator spec
        torch_device = torch.device("cpu")
        if device == Device.GPU and torch.cuda.is_available():
            torch_device = torch.device("cuda")
        if weight_dtype is not None and torch_device.type == "cpu" and weight_dtype in (torch.float16, torch.bfloat16):
            logger.info(
                "OnnxDiscrepancyCheck skipping reference model cast to %s on CPU because the dtype is not supported.",
                weight_dtype,
            )
            ref_model = ref_model.to(torch_device)
        elif weight_dtype is not None:
            ref_model = ref_model.to(device=torch_device, dtype=weight_dtype)
            logger.info(
                "OnnxDiscrepancyCheck casting reference model weights to %s to match the ONNX model.",
                weight_dtype,
            )
        else:
            ref_model = ref_model.to(torch_device)

        # Save reference PyTorch model for direct comparison
        report_dir = config.report_output_dir or output_model_path
        report_dir_path = Path(report_dir)
        if report_dir_path.suffix and not report_dir_path.is_dir():
            report_dir = str(report_dir_path.parent)
        if config.save_reference_model_state_dict:
            self._export_reference_model(ref_model, report_dir)

        session = model.prepare_session(
            device=device,
            execution_providers=[execution_provider] if execution_provider else None,
        )

        # Run inference on both and compare
        all_max_abs_diff = []
        all_count_above_0_1 = []
        all_count_above_0_01 = []
        total_elements = 0

        with torch.no_grad():
            for batch in dataloader:
                # Extract input data (batch may be (data, label) or just data)
                input_data = batch[0] if isinstance(batch, (tuple, list)) else batch

                # Run PyTorch inference
                if isinstance(input_data, dict):
                    torch_inputs = {k: v.clone().to(torch_device) for k, v in input_data.items()}
                else:
                    torch_inputs = input_data.to(torch_device)

                torch_output = ref_model(**torch_inputs)
                torch_logits = torch_output.logits.detach()
                # Run ONNX inference
                onnx_input_feed = format_data(input_data, io_config)
                onnx_outputs = session.run(None, onnx_input_feed)
                onnx_logits = _onnx_output_to_torch(onnx_outputs[0], torch_logits.dtype)

                # Compute element-wise differences using torch in double precision
                torch_logits = torch_logits.to(torch.float64).cpu()
                onnx_logits = onnx_logits.to(torch.float64).cpu()
                abs_diff = torch.abs(torch_logits - onnx_logits)
                all_max_abs_diff.append(float(torch.max(abs_diff)))
                all_count_above_0_1.append(int(torch.sum(abs_diff > 0.1)))
                all_count_above_0_01.append(int(torch.sum(abs_diff > 0.01)))
                total_elements += abs_diff.numel()

        max_abs_error = max(all_max_abs_diff)
        count_above_0_1 = sum(all_count_above_0_1)
        count_above_0_01 = sum(all_count_above_0_01)

        results = {
            "max_abs_error": max_abs_error,
            "elements_above_0_1": count_above_0_1,
            "elements_above_0_01": count_above_0_01,
            "total_elements": total_elements,
        }

        summary = (
            f"OnnxDiscrepancyCheck: max_abs_error={max_abs_error:.6f}, "
            f"elements_above_0.1={count_above_0_1}/{total_elements}, "
            f"elements_above_0.01={count_above_0_01}/{total_elements}"
        )
        logger.info(summary)

        # Measure inference speedup (ONNX vs PyTorch) on the target device
        if config.timing_iterations > 0:
            timing = self._measure_speedup(
                ref_model,
                session,
                dataloader,
                io_config,
                torch_device,
                config.warmup_iterations,
                config.timing_iterations,
            )
            if timing is not None:
                pytorch_time, onnx_time, speedup = timing
                results["pytorch_latency_s"] = pytorch_time
                results["onnx_latency_s"] = onnx_time
                results["speedup"] = speedup
        else:
            logger.info(
                "OnnxDiscrepancyCheck speedup measurement skipped because timing_iterations=%d.",
                config.timing_iterations,
            )

        # Check thresholds
        failures = []
        if config.max_mae is not None and max_abs_error > config.max_mae:
            failures.append(f"Max absolute error {max_abs_error:.6f} exceeds threshold {config.max_mae:.6f}")
        if config.max_elements_above_0_1 is not None and count_above_0_1 > config.max_elements_above_0_1:
            failures.append(
                f"Elements with diff > 0.1: {count_above_0_1} exceeds threshold {config.max_elements_above_0_1}"
            )
        if config.max_elements_above_0_01 is not None and count_above_0_01 > config.max_elements_above_0_01:
            failures.append(
                f"Elements with diff > 0.01: {count_above_0_01} exceeds threshold {config.max_elements_above_0_01}"
            )

        if failures:
            results["status"] = "failed"
            results["failures"] = failures
            failure_msg = "ONNX model discrepancy check FAILED:\n" + "\n".join(f"  - {f}" for f in failures)
            logger.error(failure_msg)
        else:
            results["status"] = "passed"

        # Generation token sequence comparison (transformers vs ONNX Runtime GenAI)
        if config.genai_model_path:
            gen_results = self.compare_generation(config, ref_model)
            longest_common = gen_results["longest_common_token_sequence"]
            results.update(gen_results)
            results["genai_model_path"] = config.genai_model_path
            if config.min_longest_common_tokens is not None and longest_common < config.min_longest_common_tokens:
                results["status"] = "failed"
                gen_failure = (
                    f"Longest common token sequence length {longest_common} is below "
                    f"threshold {config.min_longest_common_tokens}"
                )
                results.setdefault("failures", []).append(gen_failure)
                logger.error("ONNX model discrepancy check FAILED: %s", gen_failure)

        # Save results to disk
        report_path = Path(report_dir) / "discrepancy_check_results.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2))

        # Store results in model attributes so the CLI can persist them in the output directory
        model_attributes = dict(model.model_attributes) if model.model_attributes else {}
        model_attributes["discrepancy_check_results"] = results
        model.model_attributes = model_attributes
        return model

    def _measure_speedup(
        self, ref_model, session, dataloader, io_config, torch_device, warmup_iterations, timing_iterations
    ) -> tuple[float, float, float] | None:
        """Measure inference latencies and speedup of ONNX over PyTorch on the target device.

        Returns a tuple ``(pytorch_time, onnx_time, speedup)`` of the average PyTorch and ONNX
        per-iteration latencies (in seconds) and the ONNX-over-PyTorch speedup, or ``None`` when
        measurement is skipped.
        """
        if timing_iterations <= 0:
            logger.info(
                "OnnxDiscrepancyCheck speedup measurement skipped because timing_iterations=%d.",
                timing_iterations,
            )
            return None

        import torch

        from olive.common.utils import format_data

        # Use the first batch for timing
        first_batch = next(iter(dataloader))
        input_data = first_batch[0] if isinstance(first_batch, (tuple, list)) else first_batch

        if isinstance(input_data, dict):
            torch_inputs = {k: v.clone().to(torch_device) for k, v in input_data.items()}
        else:
            torch_inputs = input_data.to(torch_device)

        onnx_input_feed = format_data(input_data, io_config)
        use_cuda_sync = torch_device.type == "cuda"

        # Warmup PyTorch
        with torch.no_grad():
            for _ in range(warmup_iterations):
                ref_model(**torch_inputs)
            if use_cuda_sync:
                torch.cuda.synchronize()

        # Time PyTorch
        with torch.no_grad():
            if use_cuda_sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(timing_iterations):
                ref_model(**torch_inputs)
            if use_cuda_sync:
                torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start) / timing_iterations

        # Warmup ONNX
        for _ in range(warmup_iterations):
            session.run(None, onnx_input_feed)

        # Time ONNX
        start = time.perf_counter()
        for _ in range(timing_iterations):
            session.run(None, onnx_input_feed)
        onnx_time = (time.perf_counter() - start) / timing_iterations

        speedup = pytorch_time / onnx_time if onnx_time > 0 else float("inf")

        logger.info(
            "OnnxDiscrepancyCheck speedup: pytorch_avg=%.4fs, onnx_avg=%.4fs, speedup=%.2fx (device=%s)",
            pytorch_time,
            onnx_time,
            speedup,
            torch_device,
        )

        return pytorch_time, onnx_time, speedup

    def compare_generation(self, config: type[BasePassConfig], ref_model) -> dict:
        """Run generation on both transformers and GenAI and compare them.

        Returns a dict with the longest common token sequence length and the time-to-first-token
        and time-to-first-N-tokens latencies (in seconds) for both transformers and ONNX Runtime
        GenAI, where N is ``config.time_to_first_n_tokens``.
        """
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise ImportError("Please install `onnxruntime-genai` to enable generation comparison.") from exc
        from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        tokenizer = AutoTokenizer.from_pretrained(config.reference_model_path)

        max_new_tokens = config.generate_max_new_tokens
        first_n = max(1, min(config.time_to_first_n_tokens, max_new_tokens)) if max_new_tokens > 0 else 0

        # Transformers generation
        input_ids = tokenizer(config.generate_prompt, return_tensors="pt").input_ids
        import torch

        input_ids = input_ids.to(ref_model.device)
        use_cuda_sync = ref_model.device.type == "cuda"

        prompt_token_count = input_ids.shape[-1]
        transformers_latency = {"start": None, "ttft": None, "ttfn": None}

        class _TransformersLatencyStopCriteria(StoppingCriteria):
            def __call__(self, generated_ids, scores, **kwargs) -> bool:
                generated_token_count = generated_ids.shape[-1] - prompt_token_count
                if generated_token_count >= 1 and transformers_latency["ttft"] is None:
                    transformers_latency["ttft"] = time.perf_counter() - transformers_latency["start"]
                if generated_token_count >= first_n and transformers_latency["ttfn"] is None:
                    transformers_latency["ttfn"] = time.perf_counter() - transformers_latency["start"]
                return False

        with torch.no_grad():
            if use_cuda_sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            transformers_latency["start"] = start
            transformers_output = ref_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([_TransformersLatencyStopCriteria()]),
            )
            if use_cuda_sync:
                torch.cuda.synchronize()
            transformers_elapsed = time.perf_counter() - start
        if max_new_tokens > 0:
            transformers_ttft = transformers_latency["ttft"] or transformers_elapsed
            transformers_ttfn = transformers_latency["ttfn"] or transformers_elapsed
        else:
            transformers_ttft = None
            transformers_ttfn = None
        transformers_tokens = transformers_output[0].cpu().tolist()

        # ONNX Runtime GenAI generation
        genai_model = og.Model(config.genai_model_path)
        genai_tokenizer = og.Tokenizer(genai_model)
        genai_input_ids = genai_tokenizer.encode(config.generate_prompt)

        params = og.GeneratorParams(genai_model)
        params.set_search_options(max_length=len(genai_input_ids) + max_new_tokens, do_sample=False)

        generator = og.Generator(genai_model, params)
        generator.append_tokens([genai_input_ids])
        genai_tokens = list(genai_input_ids)
        genai_ttft = None
        genai_ttfn = None
        num_generated = 0
        start = time.perf_counter()
        while not generator.is_done():
            generator.generate_next_token()
            genai_tokens.append(generator.get_next_tokens()[0])
            num_generated += 1
            if num_generated == 1:
                genai_ttft = time.perf_counter() - start
            if num_generated == first_n:
                genai_ttfn = time.perf_counter() - start
        del generator

        longest_common = _longest_common_token_sequence(transformers_tokens, genai_tokens)

        gen_results = {
            "longest_common_token_sequence": longest_common,
            "time_to_first_n_tokens": first_n,
            "transformers_time_to_first_token_s": transformers_ttft,
            "transformers_time_to_first_n_tokens_s": transformers_ttfn,
            "genai_time_to_first_token_s": genai_ttft,
            "genai_time_to_first_n_tokens_s": genai_ttfn,
        }

        gen_summary = (
            f"OnnxDiscrepancyCheck generation comparison: "
            f"transformers_len={len(transformers_tokens)}, genai_len={len(genai_tokens)}, "
            f"longest_common_token_sequence={longest_common}, "
            f"transformers_ttft={_format_seconds(transformers_ttft)}, "
            f"transformers_time_to_first_{first_n}_tokens={_format_seconds(transformers_ttfn)}, "
            f"genai_ttft={_format_seconds(genai_ttft)}, "
            f"genai_time_to_first_{first_n}_tokens={_format_seconds(genai_ttfn)}"
        )
        logger.info(gen_summary)

        return gen_results

    def _export_reference_model(self, ref_model, output_model_path: str):
        """Save the reference PyTorch model weights for direct comparison."""
        import torch

        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        ref_pt_path = output_dir / "reference_model.pt"
        torch.save(ref_model.state_dict(), str(ref_pt_path))
        logger.info("Reference PyTorch model saved to %s", ref_pt_path)
