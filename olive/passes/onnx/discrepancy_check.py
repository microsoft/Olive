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
    return tuple(d if isinstance(d, int) else default_values[d] for d in dynamic_shape)


class OnnxDiscrepancyCheck(Pass):
    """Validates ONNX model outputs against a reference PyTorch model.

    This pass does not transform the model. It runs inference on both the
    ONNX model and a reference PyTorch/HuggingFace model with the same inputs,
    then compares outputs element-wise. It reports:
    - Maximum absolute error (MaxAE)
    - Number of elements where the absolute difference exceeds 0.1
    - Number of elements where the absolute difference exceeds 0.01

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
        else:
            raise RuntimeError(f"No data_config provided and model IO config is not static, io_config={io_config}")

        # Create dataloader
        dc = data_config.to_data_container()
        dataloader = dc.create_dataloader()

        # Load reference PyTorch model
        from transformers import AutoModelForCausalLM

        ref_model = AutoModelForCausalLM.from_pretrained(config.reference_model_path)
        ref_model.eval()

        # Prepare ONNX session
        import onnxruntime as ort

        session = ort.InferenceSession(model.model_path)
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
                torch_logits = torch_output.logits.numpy()

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

        # Return the model unchanged
        return model
