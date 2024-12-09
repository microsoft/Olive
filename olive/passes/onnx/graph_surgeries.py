# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
from typing import Any, ClassVar, Dict, List, Type

import numpy as np
import onnx
from onnx import ModelProto
from onnx.helper import make_tensor

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class Surgeon:
    registry: ClassVar[Dict[str, Type["Surgeon"]]] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Surgeon.registry[cls.__name__.lower()] = cls

    def __call__(self, model: ModelProto):
        raise NotImplementedError

    @staticmethod
    def get_node_by_name(model, name: str, match_output: bool = False):
        for node in model.graph.node:
            if (match_output and node.output[0] == name) or (not match_output and node.name == name):
                return node
        return None

    @staticmethod
    def get_tensor_shapes(model) -> Dict[str, List[int]]:
        return {info.name: [x.dim_value for x in info.type.tensor_type.shape.dim] for info in model.graph.value_info}

    @staticmethod
    def get_tensor_types(model):
        return {info.name: info.type.tensor_type.elem_type for info in model.graph.value_info}

    @staticmethod
    def get_initializer_types(model):
        return {initializer.name: initializer.data_type for initializer in model.graph.initializer}

    @staticmethod
    def get_initializer_shapes(model) -> Dict[str, List[int]]:
        return {initializer.name: initializer.dims for initializer in model.graph.initializer}

    @staticmethod
    def get_initializer_by_name(model, name: str):
        for initializer in model.graph.initializer:
            if initializer.name == name:
                return initializer
        return None


class RenameInputs(Surgeon):
    def __init__(self, old_names: List[str], new_names: List[str]):
        self.old_names = old_names
        self.new_names = new_names

    def __call__(self, model: ModelProto):
        for old_name, new_name in zip(self.old_names, self.new_names):
            for node in model.graph.node:
                for idx, input_name in enumerate(node.input):
                    if input_name == old_name:
                        node.input[idx] = new_name

                for idx, output_name in enumerate(node.output):
                    if output_name == old_name:
                        node.output[idx] = new_name

            for idx, graph_input in enumerate(model.graph.input):
                if graph_input.name == old_name:
                    model.graph.input[idx].name = new_name

            for idx, graph_output in enumerate(model.graph.output):
                if graph_output.name == old_name:
                    model.graph.output[idx].name = new_name
        return model


class RenameOutputs(Surgeon):
    def __init__(self, old_names: List[str], new_names: List[str]):
        self.old_names = old_names
        self.new_names = new_names

    def __call__(self, model: ModelProto):
        for old_name, new_name in zip(self.old_names, self.new_names):
            for node in model.graph.node:
                for idx, input_name in enumerate(node.input):
                    if input_name == old_name:
                        node.input[idx] = new_name

                for idx, output_name in enumerate(node.output):
                    if output_name == old_name:
                        node.output[idx] = new_name

            for graph_input in model.graph.input:
                if graph_input.name == old_name:
                    graph_input.name = new_name

            for graph_output in model.graph.output:
                if graph_output.name == old_name:
                    graph_output.name = new_name
        return model


class InferShapes(Surgeon):
    def __init__(self):
        pass

    def __call__(self, model: ModelProto):
        return onnx.shape_inference.infer_shapes(model)


class RemoveShapes(Surgeon):
    def __init__(self):
        pass

    def __call__(self, model: ModelProto):
        while len(model.graph.value_info) > 0:
            model.graph.value_info.pop()
        return model


class RemoveInitializerFromInputs(Surgeon):
    def __init__(self):
        pass

    def __call__(self, model: ModelProto):
        initializer_names = {initializer.name for initializer in model.graph.initializer}
        updated_inputs = [graph_input for graph_input in model.graph.input if graph_input.name not in initializer_names]
        del model.graph.input[:]
        model.graph.input.extend(updated_inputs)
        return model


class ReorderInputs(Surgeon):
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, model: ModelProto):
        inputs = list(model.graph.input)
        num_inputs = len(inputs)

        if sorted(self.permutation) != list(range(num_inputs)):
            raise ValueError("Invalid permutation: permutation must be a rearrangement of input indices.")

        reordered_inputs = [inputs[idx] for idx in self.permutation]
        del model.graph.input[:]
        model.graph.input.extend(reordered_inputs)

        return model


class ZeroOutInput(Surgeon):
    def __init__(self, node_name, input_idx):
        self.node_name = node_name
        self.input_idx = input_idx

    def __call__(self, model: ModelProto):
        from onnx.helper import make_node
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

        node = self.get_node_by_name(model, self.node_name)
        if node is None:
            logger.warning("Node %s not found in the model.", self.node_name)
            return model

        input_name = node.input[self.input_idx]

        target_shape = None
        target_type = None

        shapes = self.get_tensor_shapes(model)
        types = self.get_tensor_types(model)

        target_node = self.get_node_by_name(model, input_name, True)
        if target_node is not None:
            if input_name in shapes and input_name in types:
                target_shape = shapes[input_name]
                target_type = types[input_name]
            else:
                logger.warning("Cannot determine shape and type for input '%s'.", input_name)
                return model

        elif input_name in {inp.name for inp in model.graph.input}:
            target = next(inp for inp in model.graph.input if inp.name == input_name)
            if target.type.tensor_type.shape.dim:
                target_shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in target.type.tensor_type.shape.dim]
                target_type = target.type.tensor_type.elem_type
            else:
                logger.warning("Cannot determine shape and type for input '%s'.", input_name)
                return model

        elif input_name in {init.name for init in model.graph.initializer}:
            target = next(init for init in model.graph.initializer if init.name == input_name)
            target_shape = target.dims
            target_type = target.data_type
        else:
            logger.warning("Input '%s' not found in the model.", input_name)
            return model

        if target_shape is None or target_type is None:
            raise ValueError(f"Cannot determine shape and type for input '{input_name}'.")

        zero_values = np.zeros(target_shape, dtype=TENSOR_TYPE_TO_NP_TYPE[target_type])

        zero_tensor = make_tensor(
            name=f"{self.node_name}_zero_tensor",
            data_type=target_type,
            dims=target_shape,
            vals=zero_values.flatten().tolist(),
        )

        zero_node = make_node(
            op_type="Constant",
            inputs=[],
            outputs=[f"{self.node_name}_zero_output_0"],
            name=f"{self.node_name}_zero",
            value=zero_tensor,
        )

        model.graph.node.append(zero_node)
        node.input[self.input_idx] = zero_node.output[0]

        return model


class RemoveInputs(Surgeon):
    def __init__(self, names):
        self.names = names

    def __call__(self, model: ModelProto):
        for name in self.names:
            for graph_input in model.graph.input:
                if graph_input.name == name:
                    model.graph.input.remove(graph_input)
                    break

        nodes_to_remove = []

        for node in model.graph.node:
            node.input[:] = [input_name for input_name in node.input if input_name not in self.names]
            if len(node.input) == 0:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            model.graph.node.remove(node)

        return model


class ExposeOutputs(Surgeon):
    def __init__(self, names):
        self.names = names

    def __call__(self, model: ModelProto):
        for node in model.graph.node:
            if node.name in self.names:
                model.graph.output.extend([onnx.ValueInfoProto(name=node.output[0])])
        return model


class ExposeQuantizedOutput(Surgeon):
    def __init__(self, output_name):
        self.output_name = output_name

    def _make_name(self, name):
        return f"{self.output_name}_exposed_{name}"

    def _remove_output(self, model):
        idx = -1
        for i, output in enumerate(model.graph.output):
            if output.name == self.output_name:
                model.graph.output.pop(i)
                idx = i
                break
        if idx == -1:
            raise ValueError(f"Output '{self.output_name}' not found in model outputs.")
        return idx

    def _remove_node(self, model, target):
        idx = -1
        for i, node in enumerate(model.graph.node):
            if node == target:
                model.graph.node.pop(i)
                idx = i
                break
        if idx == -1:
            raise ValueError(f"Node '{target.name}' not found in model nodes.")
        return idx

    def _add_protos_to_model(self, model, initializer, node, tensor_type, tensor_shape):
        model.graph.initializer.append(initializer)
        model.graph.node.append(node)
        value_info = onnx.helper.make_tensor_value_info(name=node.output[0], elem_type=tensor_type, shape=tensor_shape)
        model.graph.output.append(value_info)
        return model

    def _add_scale(self, model, scale_value):
        name = self._make_name("scale")
        scale_array = np.array([scale_value], dtype=np.float32)
        initializer = make_tensor(
            name=f"{name}_value", data_type=onnx.TensorProto.FLOAT, dims=scale_array.shape, vals=scale_array
        )
        node = onnx.helper.make_node(
            op_type="Identity",
            name=name,
            inputs=[initializer.name],
            outputs=[f"{name}_output"],
        )
        tensor_type = onnx.TensorProto.FLOAT
        tensor_shape = scale_array.shape
        return self._add_protos_to_model(model, initializer, node, tensor_type, tensor_shape)

    def _add_zero_point(self, model, zero_point_value, onnx_dtype, np_dtype):
        name = self._make_name("zero_point")
        zero_point_array = np.array([zero_point_value], dtype=np_dtype)
        initializer = make_tensor(
            name=f"{name}_value", data_type=onnx_dtype, dims=zero_point_array.shape, vals=zero_point_array
        )
        node = onnx.helper.make_node(
            op_type="Identity", name=name, inputs=[initializer.name], outputs=[f"{name}_output"]
        )
        tensor_type = onnx_dtype
        tensor_shape = zero_point_array.shape
        return self._add_protos_to_model(model, initializer, node, tensor_type, tensor_shape)

    def __call__(self, model: ModelProto):
        from onnx.helper import tensor_dtype_to_np_dtype
        from onnx.numpy_helper import to_array

        output_idx = self._remove_output(model)

        dequantized_node = self.get_node_by_name(model, self.output_name, match_output=True)
        if dequantized_node is None:
            raise ValueError(f"Dequantized node producing output '{self.output_name}' not found.")

        _ = self._remove_node(model, dequantized_node)

        quantized_tensor_name = dequantized_node.input[0]
        quantized_node = self.get_node_by_name(model, quantized_tensor_name, match_output=True)
        if quantized_node is None:
            raise ValueError(f"Quantized node producing tensor '{quantized_tensor_name}' not found.")

        quantized_output_value_info = onnx.helper.make_tensor_value_info(
            name=quantized_node.output[0], elem_type=onnx.TensorProto.UINT8, shape=[]
        )
        model.graph.output.insert(output_idx, quantized_output_value_info)

        scale_initializer = self.get_initializer_by_name(model, quantized_node.input[1])
        if scale_initializer is None:
            raise ValueError(f"Scale initializer '{quantized_node.input[1]}' not found.")
        scale_value = to_array(scale_initializer)[0]
        model = self._add_scale(model, scale_value)

        zero_point_initializer = self.get_initializer_by_name(model, quantized_node.input[2])
        if zero_point_initializer is None:
            raise ValueError(f"Zero point initializer '{quantized_node.input[2]}' not found.")
        zero_point_value = to_array(zero_point_initializer)[0]
        zero_point_onnx_dtype = zero_point_initializer.data_type
        zero_point_np_dtype = tensor_dtype_to_np_dtype(zero_point_onnx_dtype)
        return self._add_zero_point(model, zero_point_value, zero_point_onnx_dtype, zero_point_np_dtype)


class GraphSurgeries(Pass):
    """ONNX graph surgeries collections.

    This pass applies a list of surgeries to the ONNX model.
    Each surgery is a transformation on the ONNX graph.

    Example:
        surgeries: {
            type: "GraphSurgeries",
            surgeries: [
                {
                    "surgeon": "RenameInputs",
                    "old_names": ["input1", "input2"]
                    "new_names": ["renamed_input1", "renamed_input2"]
                },
                {
                    "surgeon": "RenameOutputs",
                    "old_names": ["output1", "output2"]
                    "new_names": ["renamed_output1", "renamed_output2"]
                }
            ]
        }

    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "surgeries": PassConfigParam(
                type_=List[Dict[str, Any]],
                default_value=[],
                required=True,
                description="List of surgeries to apply, each with its type and parameters",
            ),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        surgeries = config["surgeries"]
        onnx_model = model.load_model()
        for surgery in surgeries:
            logger.info("Applying surgery: %s", surgery)
            surgeon_instance = self.init_surgeon_instance(surgery)
            onnx_model = surgeon_instance(onnx_model)
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    def init_surgeon_instance(self, surgery):
        surgeon_name = surgery.get("surgeon").lower()
        if not surgeon_name:
            raise ValueError("Surgeon is not specified")

        surgeon_class = Surgeon.registry.get(surgeon_name)
        if not surgeon_class:
            raise ValueError(f"Surgeon '{surgeon_name}' does not exist. Available surgeons: {Surgeon.registry.keys()}")

        required_params = self.get_surgeon_parameters(surgeon_class)
        provided_params = set(surgery.keys()) - {"surgeon"}
        missing_params = set(required_params) - provided_params
        extra_params = provided_params - set(required_params)

        if missing_params:
            raise ValueError(f"Missing parameters for surgery '{surgeon_name}': {missing_params}")
        if extra_params:
            raise ValueError(f"Ignoring extra parameters for surgery '{surgeon_name}': {extra_params}")

        init_params = {param: surgery[param] for param in required_params}
        return surgeon_class(**init_params)

    @staticmethod
    def get_surgeon_parameters(surgeon_class):
        signature = inspect.signature(surgeon_class.__init__)
        params = list(signature.parameters.keys())
        params.remove("self")
        return params
