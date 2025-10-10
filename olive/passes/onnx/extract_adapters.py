# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

from olive.common.utils import WeightsFileFormat, save_weights
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    DORA_NAME_PATTERNS,
    LOHA_NAME_PATTERNS,
    LORA_NAME_PATTERNS,
    AdapterType,
    get_adapter_name,
    get_external_data_config,
    model_proto_to_olive_model,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExtractAdapters(Pass):
    """Extract adapter weights from ONNX model and save them as external weights file.

    If make_inputs is False, model proto is invalid after this pass as the adapter weights point to non-existent
    external files. Inference session must be created by first loading the adapter weights using
    SessionOptions.add_external_initializers.

    If make_inputs is True, the adapter weights are inputs to the model and must be provided during inference.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "adapter_type": PassConfigParam(
                type_=AdapterType,
                default_value=AdapterType.LORA,
                description=f"Type of adapter to extract. Valid values are {AdapterType.__members__.values()}.",
            ),
            "make_inputs": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Convert adapter weights to inputs. If false, the adapter weights will be set as initializers with"
                    " external data."
                ),
            ),
            "dynamic_lora_r": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Whether the model uses dynamic shape for lora_r. Only used if make_inputs is True. Valid only for"
                    " float modules."
                ),
            ),
            "optional_inputs": PassConfigParam(
                type_=bool,
                default_value=True,
                description=(
                    "Create default initializers (empty tensor with lora_r dimension set to 0) for the adapter weights,"
                    " if inputs not provided during inference. Only used if make_inputs is True. Valid only for float"
                    " modules."
                ),
            ),
            "save_format": PassConfigParam(
                type_=WeightsFileFormat,
                default_value=WeightsFileFormat.ONNX_ADAPTER,
                description="Format to save the weights in.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        ir_model = model.load_ir_model()
        ir.external_data.load_to_model(ir_model)

        # dictionary to store adapter weights
        weights = {}

        if config.adapter_type in [AdapterType.LORA, AdapterType.DORA, AdapterType.LOHA]:
            weights = self._extract_adapter(ir_model, adapter_type=config.adapter_type)
        else:
            raise ValueError(f"Unsupported adapter type: {config.adapter_type}")

        if not weights:
            logger.info("No %s modules found in the model. Returning the original model.", config.adapter_type)
            return model

        if config.make_inputs:
            # create inputs for the weights
            for weight_name in weights:
                self._convert_initializer_to_input(ir_model, weight_name)
                self._make_dynamic_optional(ir_model, weights, weight_name, config)

        # save the weights
        weights_path = save_weights(weights, Path(output_model_path).parent / "adapter_weights", config.save_format)

        # save the model
        output_model = model_proto_to_olive_model(
            ir.to_proto(ir_model),
            output_model_path,
            config,
            external_initializers_file_name=weights_path.name if not config.make_inputs else None,
            constant_inputs_file_name=weights_path.name if config.make_inputs else None,
        )
        output_model.model_attributes = deepcopy(model.model_attributes) or {}
        # add adapter weights to the model attributes
        output_model.model_attributes["additional_files"] = additional_files = output_model.model_attributes.get(
            "additional_files", []
        )
        additional_files.append(str(weights_path))
        # save information about the weights in the model attributes
        weights_info = {name: [list(value.shape), str(value.dtype)] for name, value in weights.items()}
        if not config.make_inputs:
            output_model.model_attributes["external_initializers"] = weights_info
        else:
            output_model.model_attributes["constant_inputs"] = weights_info
        return output_model

    def _convert_initializer_to_input(self, model: ir.Model, initializer_name: str):
        """Convert a specific initializer to an input."""
        graph = model.graph

        # Check if the initializer exists
        if initializer_name not in graph.initializers:
            raise ValueError(f"Initializer '{initializer_name}' not found in graph")

        # Get the initializer
        initializer = graph.initializers[initializer_name]

        # Check if it's already an input
        if initializer in graph.inputs:
            return  # Already an input

        # Add to inputs
        graph.inputs.append(initializer)

    def _decompose_gemm(self, ir_model: ir.Model):
        """Decompose Gemm nodes into MatMul and Add nodes."""
        from onnxscript import rewriter
        from onnxscript.rewriter.rules.common import gemm_to_matmul_add_rule

        return rewriter.rewrite(ir_model, pattern_rewrite_rules=[gemm_to_matmul_add_rule])

    def _extract_adapter(self, ir_model: ir.Model, adapter_type: AdapterType = AdapterType.LORA):
        """Extract adapter weights for LoRA, DoRA, or LoHa from an ONNX model.

        LoRA:
        lora_A -> MatMul -> ...
        lora_B -> MatMul -> ...

        DoRA:
        Besides LoRA A and LoRA B, DoRA also has a learnable magnitude vector M (dora_M):
                         W' = mV + dV = mV + mAB
        AB + dora_M -> Div -> ...

        LoHa:
        hada_w1_a + hada_w1_b -> MatMul -> ...
        hada_w2_a + hada_w2_b -> MatMul -> ...
        """
        if adapter_type == AdapterType.DORA:
            ir_model = self._decompose_gemm(ir_model)

        # dictionary to store adapter weights
        weights = {}

        # Get the appropriate patterns for the adapter type
        patterns = None
        if adapter_type == AdapterType.LORA:
            patterns = LORA_NAME_PATTERNS
        elif adapter_type == AdapterType.DORA:
            patterns = DORA_NAME_PATTERNS
        elif adapter_type == AdapterType.LOHA:
            patterns = LOHA_NAME_PATTERNS
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

        to_rename = []
        for initializer in ir_model.graph.initializers.values():
            adapter_weight = get_adapter_name(initializer, patterns)
            if adapter_weight is None:
                continue

            to_rename.append((initializer, adapter_weight))

        for initializer, adapter_weight in to_rename:
            old_name = initializer.name

            # Store the weight data
            weights[adapter_weight] = initializer.const_value.numpy()

            # Rename the initializer
            initializer.name = adapter_weight

            # Update the initializers dictionary
            if old_name in ir_model.graph.initializers:
                ir_model.graph.initializers.pop(old_name)
                ir_model.graph.initializers[adapter_weight] = initializer

            # Create external tensor
            external_tensor = ir.ExternalTensor(
                location="dummy-location.bin",
                offset=None,
                length=None,
                dtype=initializer.const_value.dtype,
                shape=initializer.const_value.shape,
                name=adapter_weight,
                base_dir="",
            )

            initializer.const_value = external_tensor

        return weights

    def _make_dynamic_optional(
        self, model: ir.Model, weights: dict[str, "NDArray"], name: str, config: type[BasePassConfig]
    ):
        """Make the input dynamic and optional."""
        if "lora_magnitude_vector" in name:
            # magnitude vector's shape is independent of lora_r, so we do nothing
            return

        # Determine which dimension should be made dynamic based on pattern in name
        dim_idx = 1
        if "lora_A" in name:
            dim_idx = 1
        elif "lora_B" in name:
            dim_idx = 0
        elif "hada_w1_a" in name or "hada_w2_a" in name:
            dim_idx = 0  # For the first matrix in Hadamard products
        elif "hada_w1_b" in name or "hada_w2_b" in name:
            dim_idx = 1  # For the second matrix in Hadamard products

        # make the input dynamic
        if config.dynamic_lora_r:
            self._make_input_dim_dynamic(model, name, dim_idx, "lora_r")

        # create default initializer with the lora_r dimension set to 0
        if config.optional_inputs:
            shape = list(weights[name].shape)
            shape[dim_idx] = 0
            zero_array = np.zeros(shape, dtype=weights[name].dtype)
            initializer_value = model.graph.initializers[name]
            initializer_value.const_value = ir.Tensor(zero_array)
            model.graph.inputs.append(initializer_value)

    def _make_input_dim_dynamic(self, model: ir.Model, input_name: str, dim_idx: int, dim_param: str):
        """Make a dimension of an input dynamic."""
        # Find the input value
        input_value = None
        for inp in model.graph.inputs:
            if inp.name == input_name:
                input_value = inp
                break

        if input_value is None:
            raise ValueError(f"{input_name} is not an input.")

        if input_value.shape is None:
            raise ValueError(f"Input {input_name} does not have shape information.")

        if dim_idx >= len(input_value.shape):
            raise ValueError(
                f"Input {input_name} has rank {len(input_value.shape.dims)} but trying to access dim {dim_idx}."
            )

        # Create new shape with symbolic dimension
        new_dims = list(input_value.shape)
        if isinstance(new_dims[dim_idx], ir.SymbolicDim) and new_dims[dim_idx].value is not None:
            raise ValueError(f"Can't replace existing dynamic dim {new_dims[dim_idx].value} with {dim_param}")

        new_dims[dim_idx] = ir.SymbolicDim(dim_param)
        input_value.shape = ir.Shape(new_dims)
