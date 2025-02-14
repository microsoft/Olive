# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
from logging import getLogger
from pathlib import Path
from typing import Dict, Type, Union

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes.olive_pass import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_file
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = getLogger(__name__)


class MixedPrecisionOverrides(Pass):
    """Qnn mixed precision overrides pass.

    Pre-processes the model for mixed precision quantization by resolving
    constraints that each operator has when being converted to QNN operator
    Constraints refer to situations where certain tensor cannot be quantized
    to 16 bits standalone but rather neighboring tensors as well in order
    to have valid operators.

    Specific problem that arises here is the situation where certain tensor
    can be input to multiple nodes and each node requires different precision

    NOTE: This pass handles just initializer tensors as activation tensors are handled by onnxruntime
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "overrides_config": PassConfigParam(
                type_=Union[str, Dict],
                required=True,
                description="Path/Dict to mixed precision overrides json, with the format of {tensor_name: quant_type}",
            ),
            "element_wise_binary_ops": PassConfigParam(
                type_=list,
                default_value=None,
                required=False,
                description="List of element wise binary ops, if not provided defaults to ['Add', 'Sub', 'Mul', 'Div']",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        """Run for config.

        Pre-processes the model for mixed precision quantization by
        resolving constraints that each operator has when being converted
        to QNN operator. Constraints refer to situations where certain
        tensor cannot be quantized to 16 bits standalone but rather
        neighboring tensors as well in order to have valid operators

        Specific problem that arises here is the situation where certain
        tensor can be input to multiple nodes and each node requires
        different precision

        NOTE: This function handles just initializer tensors as activation
        tensors are handled by onnxruntime
        args:
            model: ONNXModelHandler
                ONNXModelHandler object
            config: Dict[str, Any]
                Configuration for the pass
            output_model_path: str
                Output model path
        """
        import onnx
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.onnx_model import ONNXModel

        overrides_content = {}
        if isinstance(config.overrides_config, dict):
            overrides_content = config.overrides_config
        elif isinstance(config.overrides_config, str):
            overrides_config_path = Path(config.overrides_config)
            with overrides_config_path.open() as f:
                overrides_content = json.load(f)
        else:
            raise ValueError("Invalid type for overrides_config")

        activations_16bit = [
            tensor
            for tensor in overrides_content
            if QuantType.from_string(overrides_content[tensor]) == QuantType.QUInt16
        ]
        onnx_model = model.load_model()

        overrides = {tensor: [{"quant_type": QuantType.QUInt16}] for tensor in activations_16bit}  # regular overrides
        conflict_data = {}

        init_map = {init.name: init for init in onnx_model.graph.initializer}

        def add_initializer_tensor_to_16bit_overrides(tensor_name: str):
            # Assigns 16 bit quantization to initializer.
            overrides[tensor_name] = [{"quant_type": QuantType.QUInt16}]

        def handle_conflict(tensor_name, node) -> bool:
            # Handles the situation where tensor is input to multiple nodes and each node requires different precision
            # **This function gets called only when tensor needs to be converted**

            conflict_found = False
            if tensor_name not in conflict_data:
                # check if tensor is input to multiple nodes
                input_to_nodes = self.tensor_input_to_which_nodes(onnx_model, tensor_name)
                # here we count how many nodes require 16 bit conversion
                if len(input_to_nodes) > 1:
                    conflict_found = True
                    conflict_data[tensor_name] = {}
                    conflict_data[tensor_name]["counts"] = len(input_to_nodes) - 1
                    conflict_data[tensor_name]["nodes"] = [node]
                    conflict_data[tensor_name]["node_names"] = [node.name]
            else:
                conflict_found = True
                if node.name not in conflict_data[tensor_name]["node_names"]:
                    conflict_data[tensor_name]["counts"] -= 1
                    conflict_data[tensor_name]["nodes"].append(node)
                    conflict_data[tensor_name]["node_names"].append(node.name)

            return conflict_found

        # Loop through each node and perform analysis based on operator type
        # If certain initializer tensor makes conflict, we do not convert it, but rather add it to conflict_data
        # which we analyze later

        element_wise_binary_ops = config.element_wise_binary_ops or ["Add", "Sub", "Mul", "Div"]
        for node in onnx_model.graph.node:

            if node.op_type in element_wise_binary_ops:
                # For ElementWiseBinaryOps inputs and outputs should be of same type
                if node.output[0] in activations_16bit:
                    for i in node.input:
                        if i not in activations_16bit and i in init_map and not handle_conflict(i, node):
                            add_initializer_tensor_to_16bit_overrides(i)

                elif node.input[0] in activations_16bit:
                    if (
                        node.input[1] not in activations_16bit
                        and node.input[1] in init_map
                        and not handle_conflict(node.input[1], node)
                    ):
                        add_initializer_tensor_to_16bit_overrides(node.input[1])

                elif (
                    node.input[1] in activations_16bit
                    and node.input[0] not in activations_16bit
                    and node.input[0] in init_map
                    and not handle_conflict(node.input[0], node)
                ):
                    add_initializer_tensor_to_16bit_overrides(node.input[0])

        model_modified = False
        for tensor_name, tensor_attrs in conflict_data.items():
            if tensor_attrs["counts"] == 0:
                # If initializer tensor is input to multiple nodes but all of them have 16bit constraint
                # convert the tensor regularly
                add_initializer_tensor_to_16bit_overrides(tensor_name)
            else:
                # If not then we create copy of the initializer that will be converted to 16 bits while
                # original one will be quantized to 8 bits
                copy_of_initializer = onnx.TensorProto()
                copy_of_initializer.CopyFrom(init_map[tensor_name])
                copy_of_initializer.name = copy_of_initializer.name + "_MixedPrecision_copy"
                onnx_model.graph.initializer.extend([copy_of_initializer])

                for node in tensor_attrs["nodes"]:
                    for i, item in enumerate(node.input):
                        # for all nodes that require their input to be 16 bit, we replace input tensor with copy
                        node.input[i] = copy_of_initializer.name if item == tensor_name else item
                # and assign convert operator to it
                add_initializer_tensor_to_16bit_overrides(copy_of_initializer.name)
                model_modified = True

        if not model_modified:
            logger.info("Model %s does not have any mixed precision conflicts", model.model_path)
            return model

        onnx_model = ONNXModel(onnx_model)
        onnx_model.topological_sort()
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        model_proto_to_file(
            onnx_model.model,
            output_model_path,
            save_as_external_data=config.save_as_external_data,
            all_tensors_to_one_file=config.all_tensors_to_one_file,
            external_data_name=config.external_data_name,
            size_threshold=config.size_threshold,
            convert_attribute=config.convert_attribute,
        )

        overrides_jsonable = {
            tensor: [{"quant_type": quant["quant_type"].name} for quant in quant_list]
            for tensor, quant_list in overrides.items()
        }
        model_attributes = model.model_attributes or {}
        model_attributes.update({"mixed_precision_overrides": overrides_jsonable})
        return ONNXModelHandler(output_model_path, model_attributes=model_attributes)

    # check if tensor is input to other nodes
    def tensor_input_to_which_nodes(self, onnx_model, tensor_name):
        # Returns dictionary of nodes that have tensor_name as input
        nodes = {}
        for node in onnx_model.graph.node:
            if tensor_name in node.input:
                nodes[node.name] = node
        return nodes
