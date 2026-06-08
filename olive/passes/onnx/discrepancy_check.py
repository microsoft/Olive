# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Optional

from olive.data.config import DataConfig
from olive.hardware import AcceleratorSpec
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


class OnnxDiscrepancyCheck(Pass):
    """Validates ONNX model outputs against a reference PyTorch model.

    This pass does not transform the model. It runs inference on both the
    ONNX model and a reference PyTorch/HuggingFace model with the same inputs,
    then compares outputs element-wise. It reports:
    - Maximum absolute error (MaxAE)
    - Number of elements where the absolute difference exceeds 0.1
    - Number of elements where the absolute difference exceeds 0.01
    - Longest common token sequence from the beginning between transformers
      generate and ONNX Runtime GenAI generate (when enabled)

    The pass fails if any configured threshold is exceeded.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "reference_model_path": PassConfigParam(
                type_=str,
                required=True,
                description="Path to the reference PyTorch/HuggingFace model to compare against.",
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
        import numpy as np
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

        # Prepare ONNX session
        session = model.prepare_session()
        io_config = model.io_config

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
                    torch_inputs = {k: v.clone() for k, v in input_data.items()}
                else:
                    torch_inputs = input_data

                torch_output = ref_model(**torch_inputs)
                torch_logits = torch_output.logits.detach().cpu().numpy()
                # Run ONNX inference
                onnx_input_feed = format_data(input_data, io_config)
                onnx_outputs = session.run(None, onnx_input_feed)
                onnx_logits = onnx_outputs[0]

                # Compute element-wise differences
                abs_diff = np.abs(torch_logits.astype(np.float64) - onnx_logits.astype(np.float64))
                all_max_abs_diff.append(float(np.max(abs_diff)))
                all_count_above_0_1.append(int(np.sum(abs_diff > 0.1)))
                all_count_above_0_01.append(int(np.sum(abs_diff > 0.01)))
                total_elements += abs_diff.size

        max_abs_error = max(all_max_abs_diff)
        count_above_0_1 = sum(all_count_above_0_1)
        count_above_0_01 = sum(all_count_above_0_01)

        logger.info(
            "OnnxDiscrepancyCheck: max_abs_error=%.6f, elements_above_0.1=%d/%d, elements_above_0.01=%d/%d",
            max_abs_error,
            count_above_0_1,
            total_elements,
            count_above_0_01,
            total_elements,
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
            raise RuntimeError("ONNX model discrepancy check failed:\n" + "\n".join(f"  - {f}" for f in failures))

        # Generation token sequence comparison (transformers vs ONNX Runtime GenAI)
        if config.genai_model_path:
            longest_common = self.compare_generation(config, ref_model)
            if config.min_longest_common_tokens is not None and longest_common < config.min_longest_common_tokens:
                raise RuntimeError(
                    f"ONNX model discrepancy check failed:\n"
                    f"  - Longest common token sequence length {longest_common} is below "
                    f"threshold {config.min_longest_common_tokens}"
                )

        # Return the model unchanged
        return model

    def compare_generation(self, config: type[BasePassConfig], ref_model) -> int:
        """Run generation on both transformers and GenAI, return longest common token sequence length."""
        try:
            import onnxruntime_genai as og
        except ImportError as exc:
            raise ImportError("Please install `onnxruntime-genai` to enable generation comparison.") from exc
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.reference_model_path)

        # Transformers generation
        input_ids = tokenizer(config.generate_prompt, return_tensors="pt").input_ids
        import torch

        with torch.no_grad():
            transformers_output = ref_model.generate(
                input_ids,
                max_new_tokens=config.generate_max_new_tokens,
                do_sample=False,
            )
        transformers_tokens = transformers_output[0].tolist()

        # ONNX Runtime GenAI generation
        genai_model = og.Model(config.genai_model_path)
        genai_tokenizer = og.Tokenizer(genai_model)
        genai_input_ids = genai_tokenizer.encode(config.generate_prompt)

        params = og.GeneratorParams(genai_model)
        params.set_search_options(max_length=len(genai_input_ids) + config.generate_max_new_tokens, do_sample=False)

        generator = og.Generator(genai_model, params)
        generator.append_tokens([genai_input_ids])
        genai_tokens = list(genai_input_ids)
        while not generator.is_done():
            generator.generate_next_token()
            genai_tokens.append(generator.get_next_tokens()[0])
        del generator

        longest_common = _longest_common_token_sequence(transformers_tokens, genai_tokens)

        logger.info(
            "OnnxDiscrepancyCheck generation comparison: "
            "transformers_len=%d, genai_len=%d, longest_common_token_sequence=%d",
            len(transformers_tokens),
            len(genai_tokens),
            longest_common,
        )

        return longest_common
