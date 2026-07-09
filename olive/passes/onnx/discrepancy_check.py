# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import subprocess
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


def _json_sanitize(obj):
    """Recursively convert numpy scalars/arrays to native Python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {key: _json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(item) for item in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _infer_shape(dynamic_shape, known_values=None):
    # Use an empty past-KV cache (past_sequence_length=0) so the discrepancy check is a clean
    # prefill comparison.  The dummy dataloader passes ``past_key_values.<i>.key/value`` tensors,
    # but HuggingFace ``forward`` does not accept those dotted names as keyword arguments and
    # silently drops them, so the reference model would run without a cache while the ONNX model
    # would consume a (bogus, all-ones) cache -- producing a large, meaningless discrepancy.
    # Keeping the past length at 0 makes both models perform the same prefill over ``input_ids``.
    default_values = {
        "batch_size": 1,
        "past_sequence_length": 0,
        "sequence_length": 8,
        "total_sequence_length": 8,
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


# ---------------------------------------------------------------------------
# Helper script executed inside the ``llama_env`` virtual environment.
# All llama-cpp-python / gguf imports are intentionally isolated to this
# subprocess so the main Olive process does not require those packages.
# ---------------------------------------------------------------------------
_LLAMA_CPP_HELPER_SCRIPT = '''\
"""llama.cpp inference helper for OnnxDiscrepancyCheck.

This script runs inside the llama_env virtual environment via subprocess.
It measures first-token latency using llama-cpp-python on a pre-converted GGUF file.
Results are written as a JSON object to stdout.

GGUF conversion is done separately via the convert_hf_to_gguf.py CLI from llama.cpp
before this script is invoked.
"""
import argparse
import json
import time


def run_inference(gguf_path, prompt_tokens, max_new_tokens, first_n):
    """Run greedy generation with llama.cpp and return first-token latency metrics."""
    from llama_cpp import Llama

    n_ctx = max(512, len(prompt_tokens) + max_new_tokens + 64)
    llm = Llama(model_path=gguf_path, n_ctx=n_ctx, verbose=False)

    generated = []
    ttft = None
    ttfn = None
    first_token_id = None

    # warmup
    start = time.perf_counter()
    for _, _token in zip(range(3), llm.generate(prompt_tokens, top_k=1, temp=0.0, reset=True)):
        break
    warmup_time = time.perf_counter() - start
    start = time.perf_counter()
    for token in llm.generate(prompt_tokens, top_k=1, temp=0.0, reset=True):
        count = len(generated) + 1
        if count == 1:
            ttft = time.perf_counter() - start
            first_token_id = int(token)
        if count == first_n and ttfn is None:
            ttfn = time.perf_counter() - start
        generated.append(int(token))
        if count >= max_new_tokens:
            break

    total_time = time.perf_counter() - start

    return {
        "first_token_id": first_token_id,
        "generated_tokens": generated,
        "ttft": ttft,
        "ttfn": ttfn,
        "total_time": total_time,
        "warmup_time": warmup_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama.cpp inference helper")
    parser.add_argument("--gguf_path", required=True)
    parser.add_argument("--prompt_tokens", required=True, help="JSON-encoded list of token IDs")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--first_n", type=int, default=5)
    args = parser.parse_args()

    prompt_tokens = json.loads(args.prompt_tokens)
    result = run_inference(args.gguf_path, prompt_tokens, args.max_new_tokens, args.first_n)
    print(json.dumps(result))
'''


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

    For encoder-decoder speech models (e.g. Whisper) the ONNX model is a composite
    (separate encoder/decoder graphs) rather than a single ``input_ids -> logits`` graph,
    so the single-graph logits/MAE/speedup comparison does not apply. In that case the
    pass runs only the generation comparison (transformers ``generate(input_features)``
    vs ONNX Runtime GenAI audio transcription).
    """

    # Speech encoder-decoder models (e.g. Whisper) are exported as composite models
    # (encoder.onnx + decoder.onnx). Accept the composite as a whole so the pass can run
    # the audio-based generation comparison instead of being applied per component.
    _accepts_composite_model: bool = True

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
                    "Directory where discrepancy check results are saved. "
                    "If not specified, results are written to the pass cache directory."
                ),
            ),
            "test_metrics": PassConfigParam(
                type_=Optional[list[str]],
                default_value=None,
                description=(
                    "List of test metrics to evaluate. Accepted values are ``'mae'`` (max absolute error "
                    "between ONNX and reference PyTorch outputs), ``'speedup'`` (ONNX-vs-PyTorch "
                    "inference latency), ``'first_token_20'`` (first generated token comparison over a "
                    "20-token generation between ONNX Runtime GenAI and transformers), ``'tft'`` (time to "
                    "the first generated token) and ``'tf5t'`` (time to the first 5 generated tokens). "
                    "When set, this field takes precedence over ``timing_iterations`` "
                    "and ``max_mae``: ``'speedup'`` enables timing, ``'mae'`` enforces the MAE threshold, and "
                    "the generation metrics run the transformers-vs-GenAI comparison. "
                    "Example: ``['mae', 'speedup']``. Set by the CLI ``--test_metrics`` option."
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
            "speech_audio_path": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description=(
                    "Path to an audio file (e.g. a 16 kHz mono ``.wav``) used as the input for the "
                    "generation comparison of encoder-decoder speech models such as Whisper. When "
                    "not set, a short synthetic audio signal is generated in-code. Ignored for "
                    "text/causal-LM models, which use ``generate_prompt`` instead."
                ),
            ),
            "generate_max_new_tokens": PassConfigParam(
                type_=int,
                default_value=32,
                description="Maximum number of new tokens to generate for the token sequence comparison.",
            ),
            "first_n_tokens_timed": PassConfigParam(
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
            "llama_cpp": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "When True, convert the reference HuggingFace model to GGUF format using "
                    "``convert_hf_to_gguf.py`` from llama.cpp and compare inference with llama.cpp. "
                    "Measures first-token difference between llama.cpp and the reference PyTorch model "
                    "as well as latency and speedup. All llama-cpp-python operations are executed in "
                    "the ``llama_env`` virtual environment via subprocess."
                ),
            ),
            "llama_cpp_env_path": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description=(
                    "Path to the virtual environment where llama-cpp-python and "
                    "``convert_hf_to_gguf.py`` are installed. "
                    "Defaults to 'llama_env' relative to the current working directory when "
                    "``llama_cpp`` is True. Create this environment and obtain the conversion "
                    "script and its dependencies with: "
                    "``python -m venv llama_env && llama_env/bin/pip install gguf safetensors "
                    "transformers sentencepiece protobuf "
                    "llama-cpp-python --extra-index-url "
                    "https://abetlen.github.io/llama-cpp-python/whl/cpu && "
                    "git clone --depth=1 --filter=blob:none --sparse "
                    "https://github.com/ggerganov/llama.cpp.git /tmp/llama_cpp_repo && "
                    "git -C /tmp/llama_cpp_repo sparse-checkout set convert_hf_to_gguf.py conversion && "
                    "cp /tmp/llama_cpp_repo/convert_hf_to_gguf.py llama_env/ && "
                    "cp -r /tmp/llama_cpp_repo/conversion llama_env/``."
                ),
            ),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        ref_model, ref_path = self._load_reference_model(model, config)
        report_dir = self._resolve_report_dir(config, output_model_path)

        # Encoder-decoder speech models (e.g. Whisper) are exported as composite encoder/decoder
        # graphs, so the single-graph logits/MAE/speedup comparison does not apply. Run only the
        # audio-based generation comparison for them.
        if self._is_speech_seq2seq(ref_model):
            logger.info(
                "OnnxDiscrepancyCheck detected an encoder-decoder speech model (%s); "
                "running the audio-based generation comparison only.",
                type(ref_model).__name__,
            )
            _, torch_device = self._resolve_devices()
            ref_model = self._cast_reference_model(ref_model, None, torch_device)
            results = self._run_speech_generation_comparison(model, config, ref_model, ref_path)
            self._save_results(model, results, report_dir)
            return model

        dataloader, io_config = self._prepare_dataloader(model)

        device, execution_provider, torch_device, weight_dtype = self._resolve_execution_device(model)
        ref_model = self._cast_reference_model(ref_model, weight_dtype, torch_device)

        session = model.prepare_session(
            device=device,
            execution_providers=[execution_provider] if execution_provider else None,
        )

        results = self._compute_logits_discrepancy(ref_model, session, dataloader, io_config, torch_device)

        effective_timing_iterations, effective_max_mae, generation_metrics = self._resolve_metric_settings(config)

        self._run_speedup_measurement(
            ref_model, session, dataloader, io_config, torch_device, config, effective_timing_iterations, results
        )

        self._check_error_thresholds(config, results, effective_max_mae)

        self._run_generation_comparison(model, config, ref_model, ref_path, generation_metrics, results)

        self._run_llama_cpp_comparison(model, config, ref_model, ref_path, report_dir, generation_metrics, results)

        self._save_results(model, results, report_dir)
        return model

    @staticmethod
    def _is_speech_seq2seq(ref_model) -> bool:
        # Speech encoder-decoder models (e.g. Whisper) take audio ``input_features`` rather than
        # text ``input_ids`` as their main input; this is the robust signal to route them to the
        # audio-based generation comparison.
        config = getattr(ref_model, "config", None)
        is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        return is_encoder_decoder and getattr(ref_model, "main_input_name", "input_ids") == "input_features"

    def _prepare_dataloader(self, model: ONNXModelHandler):
        from olive.common.config_utils import validate_config
        from olive.data.template import dummy_data_config_template
        from olive.model.config.io_config import is_io_config_static

        io_config = model.io_config
        if not io_config:
            raise RuntimeError(
                f"Model IO config is missing for {model.model_path}; cannot generate dummy inputs for discrepancy check."
            )

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

        # Create dataloader
        dc = data_config.to_data_container()
        dataloader = dc.create_dataloader()
        return dataloader, io_config

    def _load_reference_model(self, model: ONNXModelHandler, config: type[BasePassConfig]):
        # Load reference PyTorch model
        from transformers import AutoConfig, AutoModelForCausalLM

        from olive.common.hf.utils import get_model_class_from_config

        # Resolve the reference model path.  Use the configured path if it exists as a local
        # directory; otherwise fall back to a ``reference_hf_model`` directory saved alongside the
        # ONNX output.  The reference model is normally kept at ``<output_path>/reference_hf_model``
        # (written by SaveTestModelConfig / the test-model flow) and persists across engine cache
        # hits, so this fallback only triggers if the configured path has been removed.
        ref_path = config.reference_model_path
        if not Path(ref_path).is_dir():
            hf_ref_dir = (model.model_attributes or {}).get("hf_reference_model_dir", "reference_hf_model")
            fallback = Path(model.model_path).parent / hf_ref_dir
            if fallback.is_dir():
                logger.info(
                    "Reference model not found at %r; using cached copy at %r.",
                    ref_path,
                    str(fallback),
                )
                ref_path = str(fallback)
            else:
                raise RuntimeError(
                    f"Reference model directory {ref_path!r} does not exist and no cached copy was "
                    f"found at {str(fallback)!r}. Re-run the optimization workflow (olive run) to "
                    "recreate the test model."
                )

        ref_cfg = AutoConfig.from_pretrained(ref_path)
        architectures = getattr(ref_cfg, "architectures", None) or []
        is_causal_lm = any("ForCausalLM" in arch for arch in architectures)
        is_conditional_generation = getattr(ref_cfg, "is_encoder_decoder", False) or any(
            "ForConditionalGeneration" in arch for arch in architectures
        )
        if not (is_causal_lm or is_conditional_generation):
            raise ValueError(
                "OnnxDiscrepancyCheck supports HuggingFace causal language models (ForCausalLM) and "
                "encoder-decoder conditional-generation models (ForConditionalGeneration, e.g. Whisper). "
                f"Got architectures={architectures}"
            )

        # Load the reference model using the concrete class declared in its config.architectures
        # (shared with the test-model save path) rather than assuming AutoModelForCausalLM, falling
        # back to AutoModelForCausalLM only when the architecture cannot be resolved.
        # The attention implementation is baked into the reference model's config.json
        # (as ``_attn_implementation``) by the SaveTestModelConfig pass, so it is picked up
        # automatically here without needing to pass ``attn_implementation`` explicitly.
        model_class = get_model_class_from_config(ref_cfg) or AutoModelForCausalLM
        ref_model = model_class.from_pretrained(ref_path, config=ref_cfg)
        ref_model.eval()
        logger.info(
            "Loaded reference model from %s with attn_implementation=%s",
            ref_path,
            getattr(ref_cfg, "_attn_implementation", None),
        )
        return ref_model, ref_path

    def _resolve_devices(self):
        """Resolve the accelerator Device and matching torch device (independent of the ONNX model)."""
        import torch

        device = self.accelerator_spec.accelerator_type if self.accelerator_spec else None
        if device is None:
            device = Device.CPU
        elif not isinstance(device, Device):
            try:
                device = Device(str(device).lower())
            except ValueError:
                logger.warning("Unknown accelerator_type=%s; falling back to CPU.", device)
                device = Device.CPU

        torch_device = torch.device("cpu")
        if device == Device.GPU and torch.cuda.is_available():
            torch_device = torch.device("cuda")
        return device, torch_device

    def _resolve_execution_device(self, model: ONNXModelHandler):
        # Determine the floating-point dtype used by the ONNX model weights and
        # cast the reference PyTorch model to match, so the comparison uses the
        # same numeric precision for the weights on both sides.
        weight_dtype = None
        onnx_weight_dtype = _infer_onnx_weight_dtype(model.load_model())
        if onnx_weight_dtype is not None:
            weight_dtype = _onnx_dtype_to_torch(onnx_weight_dtype)
        # Prepare ONNX session on the target device (fallback to CPU)
        device, torch_device = self._resolve_devices()
        execution_provider = self.accelerator_spec.execution_provider if self.accelerator_spec else None
        return device, execution_provider, torch_device, weight_dtype

    def _cast_reference_model(self, ref_model, weight_dtype, torch_device):
        import torch

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
        return ref_model

    def _resolve_report_dir(self, config: type[BasePassConfig], output_model_path: str):
        report_dir = config.report_output_dir or output_model_path
        report_dir_path = Path(report_dir)
        if report_dir_path.suffix and not report_dir_path.is_dir():
            report_dir = str(report_dir_path.parent)
        return report_dir

    def _compute_logits_discrepancy(self, ref_model, session, dataloader, io_config, torch_device):
        import torch

        from olive.common.utils import format_data

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
        return results

    def _resolve_metric_settings(self, config: type[BasePassConfig]):
        # Resolve effective metric settings: test_metrics takes precedence when set.
        # This lets the CLI store a human-readable ["mae", "speedup"] list in the config
        # while still supporting the lower-level timing_iterations / max_mae controls for
        # advanced users and backward compatibility with older configs.
        requested_metrics = set(config.test_metrics) if config.test_metrics is not None else set()
        if config.test_metrics is not None:
            effective_timing_iterations = 5 if "speedup" in requested_metrics else 0
            effective_max_mae = 0.1 if "mae" in requested_metrics else None
        else:
            effective_timing_iterations = config.timing_iterations
            effective_max_mae = config.max_mae

        # Metrics that require running token generation (transformers vs ONNX Runtime GenAI).
        generation_metrics = requested_metrics & {"first_token_20", "tft", "tf5t"}
        return effective_timing_iterations, effective_max_mae, generation_metrics

    def _run_speedup_measurement(
        self, ref_model, session, dataloader, io_config, torch_device, config, effective_timing_iterations, results
    ):
        # Measure inference speedup (ONNX vs PyTorch) on the target device
        if effective_timing_iterations > 0:
            timing = self._measure_speedup(
                ref_model,
                session,
                dataloader,
                io_config,
                torch_device,
                config.warmup_iterations,
                effective_timing_iterations,
            )
            if timing is not None:
                pytorch_time, onnx_time, speedup = timing
                results["pytorch_latency_s"] = pytorch_time
                results["onnx_latency_s"] = onnx_time
                results["speedup"] = speedup
                logger.info(
                    "OnnxDiscrepancyCheck speedup: pytorch_latency_s=%.4f, onnx_latency_s=%.4f, speedup=%.2f",
                    pytorch_time,
                    onnx_time,
                    speedup,
                )
        else:
            logger.info(
                "OnnxDiscrepancyCheck speedup measurement skipped because timing_iterations=%d.",
                effective_timing_iterations,
            )

    def _check_error_thresholds(self, config: type[BasePassConfig], results, effective_max_mae):
        max_abs_error = results["max_abs_error"]
        count_above_0_1 = results["elements_above_0_1"]
        count_above_0_01 = results["elements_above_0_01"]

        # Check thresholds
        failures = []
        if effective_max_mae is not None and max_abs_error > effective_max_mae:
            failures.append(f"Max absolute error {max_abs_error:.6f} exceeds threshold {effective_max_mae:.6f}")
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

    def _resolve_genai_model_path(self, model, config, generation_metrics):
        """Resolve the ONNX Runtime GenAI model directory.

        Uses an explicitly configured ``genai_model_path`` when set; otherwise falls back to the
        optimized model directory when it exposes a ``genai_config.json`` (as produced by the
        ModelBuilder pass). Returns ``None`` when no GenAI model can be located.
        """
        genai_model_path = config.genai_model_path
        if genai_model_path is None:
            model_dir = Path(model.model_path)
            model_dir = model_dir if model_dir.is_dir() else model_dir.parent
            if (model_dir / "genai_config.json").is_file():
                genai_model_path = str(model_dir)
                logger.info(
                    "Using optimized ONNX model directory %s as the GenAI model for generation metrics.",
                    genai_model_path,
                )
            elif generation_metrics:
                logger.warning(
                    "Generation metrics %s requested but no genai_config.json was found in %s; skipping them.",
                    sorted(generation_metrics),
                    model_dir,
                )
        return genai_model_path

    def _surface_generation_metrics(self, config, generation_metrics, gen_results, results):
        """Merge generation-comparison results into ``results`` and record threshold failures."""
        longest_common = gen_results["longest_common_token_sequence"]
        results.update(gen_results)

        # Surface the explicitly requested named metrics for easy inspection.
        if "first_token_20" in generation_metrics:
            results["first_token_20"] = {
                "transformers_first_token": gen_results.get("transformers_first_token"),
                "genai_first_token": gen_results.get("genai_first_token"),
                "first_token_matches": gen_results.get("first_token_matches"),
                "transformers_second_token": gen_results.get("transformers_second_token"),
                "genai_second_token": gen_results.get("genai_second_token"),
                "second_token_matches": gen_results.get("second_token_matches"),
                "matching_leading_tokens": longest_common,
            }
            logger.info(
                "OnnxDiscrepancyCheck first_token_20: first_token_matches=%s (transformers=%s, genai=%s), "
                "second_token_matches=%s (transformers=%s, genai=%s), matching_leading_tokens=%s",
                gen_results.get("first_token_matches"),
                gen_results.get("transformers_first_token"),
                gen_results.get("genai_first_token"),
                gen_results.get("second_token_matches"),
                gen_results.get("transformers_second_token"),
                gen_results.get("genai_second_token"),
                longest_common,
            )
        if "tft" in generation_metrics:
            results["tft"] = {
                "transformers_s": gen_results.get("transformers_ttft_s"),
                "genai_s": gen_results.get("genai_ttft_s"),
            }
            logger.info(
                "OnnxDiscrepancyCheck tft (time to first token): transformers=%s, genai=%s",
                _format_seconds(gen_results.get("transformers_ttft_s")),
                _format_seconds(gen_results.get("genai_ttft_s")),
            )
        if "tf5t" in generation_metrics:
            results["tf5t"] = {
                "transformers_s": gen_results.get("transformers_ttfn_s"),
                "genai_s": gen_results.get("genai_ttfn_s"),
            }
            logger.info(
                "OnnxDiscrepancyCheck tf5t (time to first 5 tokens): transformers=%s, genai=%s",
                _format_seconds(gen_results.get("transformers_ttfn_s")),
                _format_seconds(gen_results.get("genai_ttfn_s")),
            )

        if config.min_longest_common_tokens is not None and longest_common < config.min_longest_common_tokens:
            results["status"] = "failed"
            gen_failure = (
                f"Longest common token sequence length {longest_common} is below "
                f"threshold {config.min_longest_common_tokens}"
            )
            results.setdefault("failures", []).append(gen_failure)
            logger.error("ONNX model discrepancy check FAILED: %s", gen_failure)

    def _run_generation_comparison(
        self, model: ONNXModelHandler, config, ref_model, ref_path, generation_metrics, results
    ):
        # Generation token sequence comparison (transformers vs ONNX Runtime GenAI).
        # Runs when an explicit genai_model_path is configured or when any generation-based
        # test metric (first_token_20 / tft / tf5t) is requested.  In the latter case the
        # optimized ONNX model directory is used as the GenAI model when it exposes a
        # genai_config.json (as produced by the ModelBuilder pass).
        if config.genai_model_path is None and not generation_metrics:
            return
        genai_model_path = self._resolve_genai_model_path(model, config, generation_metrics)
        if not genai_model_path:
            return

        # first_token_20 generates 20 tokens; tf5t measures the time to the first 5 tokens.
        gen_max_new_tokens = 20 if "first_token_20" in generation_metrics else config.generate_max_new_tokens
        gen_first_n = 5 if "tf5t" in generation_metrics else config.first_n_tokens_timed
        gen_results = self.compare_generation(
            config,
            ref_model,
            ref_model_path=ref_path,
            genai_model_path=genai_model_path,
            max_new_tokens=gen_max_new_tokens,
            first_n=gen_first_n,
        )
        results["genai_model_path"] = genai_model_path
        self._surface_generation_metrics(config, generation_metrics, gen_results, results)

    def _run_speech_generation_comparison(self, model, config, ref_model, ref_path):
        """Run the audio-based generation comparison for encoder-decoder speech models (Whisper).

        These models are exported as composite encoder/decoder graphs, so only the generation
        comparison (transformers ``generate(input_features)`` vs ONNX Runtime GenAI audio
        transcription) is applicable. Returns the results dict (also saved to disk by the caller).
        """
        _, _, generation_metrics = self._resolve_metric_settings(config)
        results = {"model_kind": "speech-seq2seq", "status": "passed"}

        genai_model_path = self._resolve_genai_model_path(model, config, generation_metrics or {"first_token_20"})
        if not genai_model_path:
            logger.warning(
                "OnnxDiscrepancyCheck: no GenAI model (genai_config.json) found for the speech model; "
                "skipping the generation comparison."
            )
            results["status"] = "skipped"
            results["skip_reason"] = "no genai_config.json found for speech model"
            return results

        gen_max_new_tokens = 20 if "first_token_20" in generation_metrics else config.generate_max_new_tokens
        gen_first_n = 5 if "tf5t" in generation_metrics else config.first_n_tokens_timed
        results["genai_model_path"] = genai_model_path
        try:
            gen_results = self._compare_generation_speech(
                config,
                ref_model,
                ref_model_path=ref_path,
                genai_model_path=genai_model_path,
                max_new_tokens=gen_max_new_tokens,
                first_n=gen_first_n,
            )
        except Exception as exc:  # pylint: disable=broad-except
            # The audio-based generation comparison is an optional diagnostic. onnxruntime-genai can
            # raise runtime errors for some Whisper builds (e.g. "Invalid output name:
            # present_key_cross_*" on a genai/model-builder version mismatch). Degrade gracefully so
            # the optimize workflow still completes and the failure is recorded in the report.
            logger.warning(
                "OnnxDiscrepancyCheck speech generation comparison could not be completed (%s). This is "
                "typically an onnxruntime-genai / model-builder version incompatibility for the Whisper "
                "GenAI model; skipping the comparison.",
                exc,
            )
            results["status"] = "skipped"
            results["skip_reason"] = f"speech generation comparison failed: {exc}"
            return results

        # For speech models every generation metric is derived from the audio comparison, so always
        # surface first_token_20 alongside any explicitly requested timing metrics.
        self._surface_generation_metrics(config, generation_metrics | {"first_token_20"}, gen_results, results)
        return results

    def _run_llama_cpp_comparison(
        self, model: ONNXModelHandler, config, ref_model, ref_path, report_dir, generation_metrics, results
    ):
        # llama.cpp comparison: convert reference model to GGUF and compare latencies
        if not config.llama_cpp:
            return
        preconverted_gguf_path = None
        if model.model_attributes:
            preconverted_gguf_path = model.model_attributes.get("reference_gguf_model_path")
        try:
            # first_token_20 restricts the comparison to a 20-token generation, mirroring the
            # transformers vs GenAI path so no more than 20 generated tokens are validated.
            gen_max_new_tokens = 20 if "first_token_20" in generation_metrics else config.generate_max_new_tokens
            llama_results = self.compare_llama_cpp(
                config,
                ref_model,
                output_dir=report_dir,
                pytorch_latency_s=results.get("pytorch_latency_s"),
                onnx_latency_s=results.get("onnx_latency_s"),
                ref_model_path=ref_path,
                preconverted_gguf_path=preconverted_gguf_path,
                max_new_tokens=gen_max_new_tokens,
            )
            results.update(llama_results)

            # Surface the llama.cpp vs transformers first-token comparison alongside the
            # transformers vs GenAI comparison when first_token_20 is requested.
            if "first_token_20" in generation_metrics:
                first_token_20 = results.setdefault("first_token_20", {})
                transformers_first_token = llama_results.get("llama_cpp_pytorch_first_token_id")
                llama_first_token = llama_results.get("llama_cpp_first_token_id")
                first_token_20.setdefault("transformers_first_token", transformers_first_token)
                first_token_20["llama_cpp_first_token"] = llama_first_token
                first_token_20["llama_cpp_first_token_matches"] = llama_results.get(
                    "llama_cpp_first_token_matches_pytorch"
                )
                first_token_20.setdefault(
                    "transformers_second_token", llama_results.get("llama_cpp_pytorch_second_token_id")
                )
                first_token_20["llama_cpp_second_token"] = llama_results.get("llama_cpp_second_token_id")
                first_token_20["llama_cpp_second_token_matches"] = llama_results.get(
                    "llama_cpp_second_token_matches_pytorch"
                )
                first_token_20["llama_cpp_matching_leading_tokens"] = llama_results.get(
                    "llama_cpp_longest_common_token_sequence"
                )
                logger.info(
                    "OnnxDiscrepancyCheck first_token_20 (llama.cpp): first_token_matches=%s "
                    "(transformers=%s, llama_cpp=%s), second_token_matches=%s (transformers=%s, llama_cpp=%s), "
                    "matching_leading_tokens=%s",
                    llama_results.get("llama_cpp_first_token_matches_pytorch"),
                    transformers_first_token,
                    llama_first_token,
                    llama_results.get("llama_cpp_second_token_matches_pytorch"),
                    llama_results.get("llama_cpp_pytorch_second_token_id"),
                    llama_results.get("llama_cpp_second_token_id"),
                    llama_results.get("llama_cpp_longest_common_token_sequence"),
                )
        except Exception as exc:
            logger.exception("OnnxDiscrepancyCheck llama.cpp comparison failed.")
            results["status"] = "failed"
            results.setdefault("failures", []).append(f"llama.cpp comparison failed: {exc}")

    def _save_results(self, model: ONNXModelHandler, results, report_dir):
        # Save results to disk
        results = _json_sanitize(results)
        report_path = Path(report_dir) / "discrepancy_check_results.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2))
        logger.info("Saved discrepancy check results to %s", report_path)

        # Store results in model attributes so the CLI can persist them in the output directory
        model_attributes = dict(model.model_attributes) if model.model_attributes else {}
        model_attributes["discrepancy_check_results"] = results
        model.model_attributes = model_attributes

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

    def _load_or_make_audio(self, config):
        """Return ``(audio_array, sample_rate)`` for the speech comparison.

        Uses ``config.speech_audio_path`` when set (read as mono); otherwise generates a short
        synthetic signal so the comparison always has an input available.
        """
        import numpy as np

        audio_path = getattr(config, "speech_audio_path", None)
        if audio_path:
            import soundfile as sf

            audio, sample_rate = sf.read(audio_path, dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                # Downmix to mono.
                audio = audio.mean(axis=1)
            logger.info("OnnxDiscrepancyCheck using speech audio from %s (sample_rate=%d).", audio_path, sample_rate)
            return np.asarray(audio, dtype=np.float32), int(sample_rate)

        # Synthetic fallback: 2 seconds of a low-amplitude 440 Hz tone at 16 kHz.
        sample_rate = 16000
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False, dtype=np.float32)
        audio = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        logger.info(
            "OnnxDiscrepancyCheck using synthetic %.1fs audio at %d Hz for the speech comparison.",
            duration_s,
            sample_rate,
        )
        return audio, sample_rate

    @staticmethod
    def _audio_to_wav_bytes(audio, sample_rate: int) -> bytes:
        import io

        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, audio, samplerate=sample_rate, format="WAV")
        return buffer.getvalue()

    @staticmethod
    def _whisper_decoder_prompt(genai_model_path: str) -> str:
        """Build the Whisper decoder prompt string, mirroring olive_evaluator's genai speech path."""
        genai_config = {}
        genai_config_path = Path(genai_model_path) / "genai_config.json"
        if genai_config_path.is_file():
            with genai_config_path.open() as f:
                genai_config = json.load(f)
        # English-only Whisper checkpoints (vocab_size=51864) use a shorter prompt without a
        # language/task selector.
        vocab_size = genai_config.get("model", {}).get("vocab_size", 51865)
        if vocab_size == 51864:
            prompt_tokens = ["<|startoftranscript|>", "<|notimestamps|>"]
        else:
            prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
        return "".join(prompt_tokens)

    @staticmethod
    def _load_speech_processor(ref_model_path: str, genai_model_path: str, ref_model):
        """Load the audio processor / feature extractor for the reference speech model.

        Prefers the reference model directory (written self-contained by SaveTestModelConfig), then
        falls back to the GenAI model directory and finally to the original model id recorded in the
        config, so an older reference directory saved without a ``preprocessor_config.json`` still
        works.
        """
        from transformers import AutoProcessor

        candidates = [ref_model_path, genai_model_path]
        original_name = getattr(getattr(ref_model, "config", None), "_name_or_path", None)
        if original_name:
            candidates.append(original_name)

        last_error = None
        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                processor = AutoProcessor.from_pretrained(candidate)
                if candidate != ref_model_path:
                    logger.info(
                        "OnnxDiscrepancyCheck loaded the speech processor from %r (reference model "
                        "directory %r did not contain one).",
                        candidate,
                        ref_model_path,
                    )
                return processor
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                logger.debug("Could not load speech processor from %r: %s", candidate, exc)
        raise RuntimeError(
            f"Could not load an audio processor/feature extractor for the speech model from any of "
            f"{candidates}. Ensure the reference model directory contains a preprocessor_config.json."
        ) from last_error

    def _compare_generation_speech(
        self,
        config: type[BasePassConfig],
        ref_model,
        *,
        ref_model_path: str,
        genai_model_path: str,
        max_new_tokens: Optional[int] = None,
        first_n: Optional[int] = None,
    ) -> dict:
        """Compare transformers vs ONNX Runtime GenAI generation for a Whisper-style speech model.

        The transformers side runs ``generate(input_features=...)`` on mel features extracted from
        the audio; the GenAI side transcribes the same audio through the multimodal processor
        (reusing the proven olive_evaluator genai-whisper flow). Returns the same-shaped dict as
        :meth:`compare_generation` (longest common leading token sequence, first/second token
        matches and latency metrics), computed over the full decoder token sequences (both of which
        start with the shared ``<|startoftranscript|>`` preamble).
        """
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise ImportError("Please install `onnxruntime-genai` to enable generation comparison.") from exc

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        max_new_tokens = config.generate_max_new_tokens if max_new_tokens is None else max_new_tokens
        first_n_config = config.first_n_tokens_timed if first_n is None else first_n
        first_n = max(1, min(first_n_config, max_new_tokens)) if max_new_tokens > 0 else 0

        audio, sample_rate = self._load_or_make_audio(config)

        # ---- transformers generation (audio mel features -> decoder tokens) ----
        processor = self._load_speech_processor(ref_model_path, genai_model_path, ref_model)
        features = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = features.input_features.to(device=ref_model.device, dtype=ref_model.dtype)
        use_cuda_sync = ref_model.device.type == "cuda"

        transformers_latency = {"start": None, "ttft": None, "ttfn": None}
        prompt_token_count = {"value": None}

        class _TransformersLatencyStopCriteria(StoppingCriteria):
            def __call__(self, generated_ids, scores, **kwargs) -> bool:
                if prompt_token_count["value"] is None:
                    # First invocation carries the decoder prompt (start-of-transcript preamble).
                    prompt_token_count["value"] = generated_ids.shape[-1] - 1
                generated_token_count = generated_ids.shape[-1] - prompt_token_count["value"]
                if generated_token_count >= 1 and transformers_latency["ttft"] is None:
                    transformers_latency["ttft"] = time.perf_counter() - transformers_latency["start"]
                if generated_token_count >= first_n and transformers_latency["ttfn"] is None:
                    transformers_latency["ttfn"] = time.perf_counter() - transformers_latency["start"]
                return False

        with torch.no_grad():
            if use_cuda_sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            ref_model.generate(input_features=input_features, max_new_tokens=3, do_sample=False)
            warmup_time = time.perf_counter() - start
            if use_cuda_sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            transformers_latency["start"] = start
            transformers_output = ref_model.generate(
                input_features=input_features,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([_TransformersLatencyStopCriteria()]),
            )
            if use_cuda_sync:
                torch.cuda.synchronize()
            transformers_elapsed = time.perf_counter() - start
        transformers_tokens = transformers_output[0].cpu().tolist()
        if max_new_tokens > 0:
            transformers_ttft = (
                transformers_latency["ttft"] if transformers_latency["ttft"] is not None else transformers_elapsed
            )
            transformers_ttfn = (
                transformers_latency["ttfn"] if transformers_latency["ttfn"] is not None else transformers_elapsed
            )
        else:
            transformers_ttft = None
            transformers_ttfn = None

        # ---- ONNX Runtime GenAI generation (audio -> decoder tokens) ----
        genai_model = og.Model(genai_model_path)
        genai_processor = genai_model.create_multimodal_processor()
        prompt = self._whisper_decoder_prompt(genai_model_path)
        audios = og.Audios.open_bytes(self._audio_to_wav_bytes(audio, sample_rate))
        inputs = genai_processor([prompt], audios=audios)

        params = og.GeneratorParams(genai_model)
        params.set_search_options(do_sample=False, max_length=max_new_tokens + 64, min_length=0, batch_size=1)
        generator = og.Generator(genai_model, params)
        generator.set_inputs(inputs)

        genai_ttft = None
        genai_ttfn = None
        num_generated = 0
        start = time.perf_counter()
        while not generator.is_done():
            generator.generate_next_token()
            num_generated += 1
            if num_generated == 1:
                genai_ttft = time.perf_counter() - start
            if num_generated == first_n:
                genai_ttfn = time.perf_counter() - start
            if num_generated >= max_new_tokens:
                break
        genai_tokens = list(generator.get_sequence(0))
        del generator

        # Both token streams begin with the shared start-of-transcript decoder preamble, so the
        # comparison is over the full decoder sequences (unlike the causal-LM path, which strips a
        # known text prompt first).
        longest_common = _longest_common_token_sequence(transformers_tokens, genai_tokens)

        transformers_first_token = transformers_tokens[0] if transformers_tokens else None
        genai_first_token = genai_tokens[0] if genai_tokens else None
        first_token_matches = transformers_first_token is not None and transformers_first_token == genai_first_token
        transformers_second_token = transformers_tokens[1] if len(transformers_tokens) > 1 else None
        genai_second_token = genai_tokens[1] if len(genai_tokens) > 1 else None
        second_token_matches = transformers_second_token is not None and transformers_second_token == genai_second_token

        gen_results = {
            "longest_common_token_sequence": longest_common,
            "first_n_tokens_timed": first_n,
            "transformers_first_token": transformers_first_token,
            "genai_first_token": genai_first_token,
            "first_token_matches": first_token_matches,
            "transformers_second_token": transformers_second_token,
            "genai_second_token": genai_second_token,
            "second_token_matches": second_token_matches,
            "transformers_ttft_s": transformers_ttft,
            "transformers_ttfn_s": transformers_ttfn,
            "genai_ttft_s": genai_ttft,
            "genai_ttfn_s": genai_ttfn,
            "transformers_warmup_s": warmup_time,
        }
        logger.info(
            "OnnxDiscrepancyCheck speech generation comparison: transformers_len=%d, genai_len=%d, "
            "longest_common_token_sequence=%d, first_token_matches=%s",
            len(transformers_tokens),
            len(genai_tokens),
            longest_common,
            first_token_matches,
        )
        return gen_results

    def compare_generation(
        self,
        config: type[BasePassConfig],
        ref_model,
        *,
        ref_model_path: str,
        genai_model_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        first_n: Optional[int] = None,
    ) -> dict:
        """Run generation on both transformers and GenAI and compare them.

        Returns a dict with the longest common token sequence length, the first-generated-token
        match between transformers and ONNX Runtime GenAI, and the time-to-first-token and
        time-to-first-N-tokens latencies (in seconds) for both, where N is ``first_n``
        (defaults to ``config.first_n_tokens_timed``).

        ``genai_model_path``, ``max_new_tokens`` and ``first_n`` override the corresponding
        config values when provided, which lets the caller request specific metrics such as
        ``first_token_20`` (20-token generation) or ``tf5t`` (first 5 tokens).
        """
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise ImportError("Please install `onnxruntime-genai` to enable generation comparison.") from exc
        from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        genai_model_path = genai_model_path if genai_model_path is not None else config.genai_model_path
        tokenizer = AutoTokenizer.from_pretrained(ref_model_path)

        max_new_tokens = config.generate_max_new_tokens if max_new_tokens is None else max_new_tokens
        first_n_config = config.first_n_tokens_timed if first_n is None else first_n
        first_n = max(1, min(first_n_config, max_new_tokens)) if max_new_tokens > 0 else 0

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
            # warmup
            start = time.perf_counter()
            ref_model.generate(input_ids, max_new_tokens=3, do_sample=False)
            warmup_time = time.perf_counter() - start
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
            transformers_ttft = (
                transformers_latency["ttft"] if transformers_latency["ttft"] is not None else transformers_elapsed
            )
            transformers_ttfn = (
                transformers_latency["ttfn"] if transformers_latency["ttfn"] is not None else transformers_elapsed
            )
        else:
            transformers_ttft = None
            transformers_ttfn = None
        transformers_tokens = transformers_output[0].cpu().tolist()

        # ONNX Runtime GenAI generation.  Feed GenAI the exact same prompt token ids produced by the
        # transformers tokenizer (including any special/BOS tokens) rather than re-encoding with the
        # GenAI tokenizer.  ``og.Tokenizer.encode`` does not add special tokens by default, so
        # re-encoding would drop the BOS token that transformers adds, giving the two models different
        # inputs and a spurious first-token mismatch even when the models are numerically identical.
        genai_model = og.Model(genai_model_path)
        genai_input_ids = input_ids[0].cpu().tolist()

        params = og.GeneratorParams(genai_model)
        params.set_search_options(max_length=len(genai_input_ids) + max_new_tokens, do_sample=False)

        genai_tokens = list(genai_input_ids)
        genai_prompt_token_count = len(genai_input_ids)
        genai_ttft = None
        genai_ttfn = None
        num_generated = 0

        # warmup
        generator = og.Generator(genai_model, params)
        generator.append_tokens([genai_input_ids])
        start = time.perf_counter()
        while not generator.is_done():
            generator.generate_next_token()
            genai_tokens.append(generator.get_next_tokens()[0])
            if len(genai_tokens) >= 3:
                break
        genai_warmup_time = time.perf_counter() - start

        generator = og.Generator(genai_model, params)
        generator.append_tokens([genai_input_ids])

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

        # Longest common leading token sequence between transformers and ONNX Runtime GenAI, measured
        # over the generated tokens only (the prompt is shared and identical since GenAI is fed the same
        # token ids).  This bounds the count by ``max_new_tokens`` so, e.g., first_token_20 never
        # validates more than 20 generated tokens.
        transformers_generated_tokens = transformers_tokens[prompt_token_count:]
        genai_generated_tokens = genai_tokens[genai_prompt_token_count:]
        longest_common = _longest_common_token_sequence(transformers_generated_tokens, genai_generated_tokens)

        # First generated token comparison (transformers vs ONNX Runtime GenAI).
        transformers_first_token = (
            transformers_tokens[prompt_token_count] if len(transformers_tokens) > prompt_token_count else None
        )
        genai_first_token = (
            genai_tokens[genai_prompt_token_count] if len(genai_tokens) > genai_prompt_token_count else None
        )
        first_token_matches = transformers_first_token is not None and transformers_first_token == genai_first_token

        # Second generated token comparison (transformers vs ONNX Runtime GenAI).
        transformers_second_token = (
            transformers_tokens[prompt_token_count + 1] if len(transformers_tokens) > prompt_token_count + 1 else None
        )
        genai_second_token = (
            genai_tokens[genai_prompt_token_count + 1] if len(genai_tokens) > genai_prompt_token_count + 1 else None
        )
        second_token_matches = transformers_second_token is not None and transformers_second_token == genai_second_token

        gen_results = {
            "longest_common_token_sequence": longest_common,
            "first_n_tokens_timed": first_n,
            "transformers_first_token": transformers_first_token,
            "genai_first_token": genai_first_token,
            "first_token_matches": first_token_matches,
            "transformers_second_token": transformers_second_token,
            "genai_second_token": genai_second_token,
            "second_token_matches": second_token_matches,
            "transformers_ttft_s": transformers_ttft,
            "transformers_ttfn_s": transformers_ttfn,
            "genai_ttft_s": genai_ttft,
            "genai_ttfn_s": genai_ttfn,
            "transformers_warmup_s": warmup_time,
            "genai_warmup_s": genai_warmup_time,
        }

        gen_summary = (
            f"OnnxDiscrepancyCheck generation comparison: "
            f"transformers_len={len(transformers_tokens)}, genai_len={len(genai_tokens)}, "
            f"longest_common_token_sequence={longest_common}, "
            f"first_token_matches={first_token_matches}, "
            f"transformers_ttft={_format_seconds(transformers_ttft)}, "
            f"transformers_time_to_first_{first_n}_tokens={_format_seconds(transformers_ttfn)}, "
            f"genai_ttft={_format_seconds(genai_ttft)}, "
            f"genai_time_to_first_{first_n}_tokens={_format_seconds(genai_ttfn)}"
        )
        logger.info(gen_summary)

        return gen_results

    @staticmethod
    def _get_llama_env_python(env_path: str) -> str:
        """Return the Python interpreter path inside the given virtual environment.

        Checks both the POSIX (``bin/python``) and Windows (``Scripts/python.exe``)
        layouts so the method works cross-platform.
        """
        env = Path(env_path)
        for candidate in (env / "bin" / "python", env / "Scripts" / "python.exe"):
            if candidate.exists():
                return str(candidate)
        raise RuntimeError(
            f"Could not find a Python interpreter in the llama_env at '{env_path}'. "
            "Create the environment with: "
            "python -m venv llama_env && llama_env/bin/pip install gguf safetensors "
            "llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
        )

    @staticmethod
    def _get_convert_script(env_path: str) -> str:
        r"""Return the path to the ``convert_hf_to_gguf.py`` conversion script.

        The script and the accompanying ``conversion/`` package must be placed at the root
        of the virtual environment directory (i.e. ``{env_path}/convert_hf_to_gguf.py`` and
        ``{env_path}/conversion/``).  Obtain them via a sparse clone::

            git clone --depth=1 --filter=blob:none --sparse \
                https://github.com/ggerganov/llama.cpp.git /tmp/llama_cpp_repo
            git -C /tmp/llama_cpp_repo sparse-checkout set convert_hf_to_gguf.py conversion
            cp /tmp/llama_cpp_repo/convert_hf_to_gguf.py {env_path}/
            cp -r /tmp/llama_cpp_repo/conversion {env_path}/
        """
        env = Path(env_path)
        script = env / "convert_hf_to_gguf.py"
        conversion_pkg = env / "conversion"
        setup_cmd = (
            f"git clone --depth=1 --filter=blob:none --sparse "
            f"https://github.com/ggerganov/llama.cpp.git /tmp/llama_cpp_repo && "
            f"git -C /tmp/llama_cpp_repo sparse-checkout set convert_hf_to_gguf.py conversion && "
            f"cp /tmp/llama_cpp_repo/convert_hf_to_gguf.py {env_path}/ && "
            f"cp -r /tmp/llama_cpp_repo/conversion {env_path}/"
        )
        if not script.exists():
            raise RuntimeError(
                f"Could not find convert_hf_to_gguf.py in '{env_path}'. "
                f"Clone it from the llama.cpp repository: {setup_cmd}"
            )
        if not conversion_pkg.exists():
            raise RuntimeError(
                f"Could not find the 'conversion' package in '{env_path}'. "
                "convert_hf_to_gguf.py requires the 'conversion/' directory alongside it. "
                f"Clone it from the llama.cpp repository: {setup_cmd}"
            )
        return str(script)

    def compare_llama_cpp(
        self,
        config: type[BasePassConfig],
        ref_model,
        output_dir: str,
        pytorch_latency_s: Optional[float] = None,
        onnx_latency_s: Optional[float] = None,
        *,
        ref_model_path: str,
        preconverted_gguf_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict:
        """Convert the reference model to GGUF and compare inference with llama.cpp.

        All llama-cpp-python operations are executed inside the ``llama_env`` virtual
        environment via subprocess, so the main Olive process does not need
        llama-cpp-python installed.

        The method:

        1. Saves the reference model and tokenizer to ``output_dir/hf_model`` using
           ``save_pretrained`` (standard HuggingFace format).
        2. Calls ``convert_hf_to_gguf.py`` from llama.cpp via the command line to
           convert the saved directory to a GGUF F32 file at ``output_dir/model.gguf``.
        3. Runs ``_LLAMA_CPP_HELPER_SCRIPT`` inside ``llama_env`` to measure
           first-token latency with llama-cpp-python on the converted GGUF file.
        4. Returns a metrics dict with the llama.cpp results and speedup ratios
           relative to PyTorch and ONNX when those latencies are provided.
        """
        import torch
        from transformers import AutoTokenizer

        # Resolve the llama_env Python interpreter and conversion script
        env_path = config.llama_cpp_env_path or "llama_env"
        python_path = self._get_llama_env_python(env_path)

        # Tokenize the generation prompt using the main-env tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ref_model_path)
        encoded = tokenizer(config.generate_prompt, return_tensors="pt")
        prompt_token_ids: list[int] = encoded["input_ids"][0].tolist()

        max_new_tokens = config.generate_max_new_tokens if max_new_tokens is None else max_new_tokens
        first_n = max(1, min(config.first_n_tokens_timed, max_new_tokens)) if max_new_tokens > 0 else 1

        # Run generation with transformers to get the reference first token and the leading
        # token sequence used for the longest-common-token comparison against llama.cpp.
        input_ids = torch.tensor([prompt_token_ids]).to(ref_model.device)
        with torch.no_grad():
            gen_out = ref_model.generate(input_ids, max_new_tokens=max(1, max_new_tokens), do_sample=False)
        pytorch_tokens: list[int] = gen_out[0].cpu().tolist()
        prompt_token_count = len(prompt_token_ids)
        pytorch_first_token_id = (
            pytorch_tokens[prompt_token_count] if len(pytorch_tokens) > prompt_token_count else None
        )
        pytorch_second_token_id = (
            pytorch_tokens[prompt_token_count + 1] if len(pytorch_tokens) > prompt_token_count + 1 else None
        )

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        model_dir = str(output_dir_path / "hf_model")
        gguf_path = str(output_dir_path / "model.gguf")
        script_path = str(output_dir_path / "llama_cpp_helper.py")

        if preconverted_gguf_path and Path(preconverted_gguf_path).exists():
            gguf_path = preconverted_gguf_path
            logger.info("Using pre-converted GGUF from %s", gguf_path)
        else:
            convert_script = self._get_convert_script(env_path)
            # Save model and tokenizer in standard HuggingFace format.
            ref_model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
            logger.info("Saved reference HuggingFace model and tokenizer to %s", model_dir)

            # Step 1: Convert to GGUF using the official convert_hf_to_gguf.py CLI.
            subprocess.run(
                [python_path, convert_script, model_dir, "--outfile", gguf_path, "--outtype", "f32"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Converted HuggingFace model to GGUF at %s", gguf_path)

        # Step 2: Run inference inside llama_env using the pre-converted GGUF file.
        (output_dir_path / "llama_cpp_helper.py").write_text(_LLAMA_CPP_HELPER_SCRIPT)

        try:
            proc = subprocess.run(
                [
                    python_path,
                    script_path,
                    "--gguf_path",
                    gguf_path,
                    "--prompt_tokens",
                    json.dumps(prompt_token_ids),
                    "--max_new_tokens",
                    str(max_new_tokens),
                    "--first_n",
                    str(first_n),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            results = {"llama_cpp_out": e.stderr, "llama_cpp_err": e.stdout}
            logger.info("OnnxDiscrepancyCheck llama.cpp error=%s output=%s", e.stderr, e.stdout)
            return results

        llama_out: dict = json.loads(proc.stdout)
        llama_first_token_id: Optional[int] = llama_out.get("first_token_id")
        llama_generated_tokens: list[int] = llama_out.get("generated_tokens") or []
        llama_second_token_id: Optional[int] = llama_generated_tokens[1] if len(llama_generated_tokens) > 1 else None
        llama_ttft: Optional[float] = llama_out.get("ttft")
        llama_ttfn: Optional[float] = llama_out.get("ttfn")
        llama_total: Optional[float] = llama_out.get("total_time")
        llama_warmup: Optional[float] = llama_out.get("warmup_time")

        # Longest common leading token sequence between transformers and llama.cpp, measured over
        # the generated tokens only (the prompt is shared and identical).  This bounds the count by
        # ``max_new_tokens`` so, e.g., first_token_20 never validates more than 20 generated tokens.
        pytorch_generated_tokens = pytorch_tokens[prompt_token_count:]
        llama_longest_common = _longest_common_token_sequence(pytorch_generated_tokens, llama_generated_tokens)

        results = {
            "llama_cpp_pytorch_first_token_id": pytorch_first_token_id,
            "llama_cpp_first_token_id": llama_first_token_id,
            "llama_cpp_first_token_matches_pytorch": llama_first_token_id == pytorch_first_token_id,
            "llama_cpp_pytorch_second_token_id": pytorch_second_token_id,
            "llama_cpp_second_token_id": llama_second_token_id,
            "llama_cpp_second_token_matches_pytorch": (
                pytorch_second_token_id is not None and llama_second_token_id == pytorch_second_token_id
            ),
            "llama_cpp_longest_common_token_sequence": llama_longest_common,
            "llama_cpp_ttft_s": llama_ttft,
            "llama_cpp_ttfn_s": llama_ttfn,
            "llama_cpp_total_time_s": llama_total,
            "llama_cpp_warmup_s": llama_warmup,
        }

        logger.info(
            "OnnxDiscrepancyCheck llama.cpp comparison: first_token_matches_pytorch=%s, "
            "matching_leading_tokens=%s, ttft=%s, ttfn=%s, total=%s",
            results["llama_cpp_first_token_matches_pytorch"],
            llama_longest_common,
            _format_seconds(llama_ttft),
            _format_seconds(llama_ttfn),
            _format_seconds(llama_total),
        )
        return results
