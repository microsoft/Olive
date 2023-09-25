# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx

TENSOR_TYPE_MAP = {
    "float": onnx.TensorProto.FLOAT,
    "uint8": onnx.TensorProto.UINT8,
    "int8": onnx.TensorProto.INT8,
    "uint16": onnx.TensorProto.UINT16,
    "int16": onnx.TensorProto.INT16,
    "int32": onnx.TensorProto.INT32,
    "int64": onnx.TensorProto.INT64,
    "string": onnx.TensorProto.STRING,
    "bool": onnx.TensorProto.BOOL,
    "float16": onnx.TensorProto.FLOAT16,
    "double": onnx.TensorProto.DOUBLE,
    "uint32": onnx.TensorProto.UINT32,
    "uint64": onnx.TensorProto.UINT64,
    "complex64": onnx.TensorProto.COMPLEX64,
    "complex128": onnx.TensorProto.COMPLEX128,
    "bfloat16": onnx.TensorProto.BFLOAT16,
}


def resolve_placeholder(model: onnx.ModelProto, param_value: dict):
    resolved_param = param_value
    arg_type = param_value.get("type")
    if arg_type == "__model_input__":
        if model is None:
            raise ValueError("model is required for __model_input__")

        input_index = param_value.get("input_index")
        if input_index is None:
            raise ValueError("input_index is required for __model_input__")

        dim_index = param_value.get("dim_index")
        if dim_index is None:
            raise ValueError("dim_index is required for __model_input__")
        resolved_param = get_graph_input_shape_dim_value(model.graph, input_index, dim_index)
    elif arg_type == "__model_output__":
        if model is None:
            raise ValueError("model is required for __model_output__")

        output_index = param_value.get("output_index")
        if output_index is None:
            raise ValueError("output_index is required for __model_output__")
        dim_index = param_value.get("dim_index")
        if dim_index is None:
            raise ValueError("dim_index is required for __model_output__")
        resolved_param = get_graph_output_shape_dim_value(model.graph, output_index, dim_index)

    return resolved_param


def get_graph_input_shape_dim_value(graph: onnx.GraphProto, input_index: int, dim_index: int) -> int:
    if abs(input_index) >= len(graph.input):
        raise ValueError(f"input_index {input_index} is out of range {len(graph.input)}")

    graph_input = graph.input[input_index]
    if abs(dim_index) >= len(graph_input.type.tensor_type.shape.dim):
        raise ValueError(f"dim_index {dim_index} is out of range {len(graph_input.type.tensor_type.shape.dim)}")
    return graph_input.type.tensor_type.shape.dim[dim_index].dim_value


def get_graph_output_shape_dim_value(graph: onnx.GraphProto, output_index: int, dim_index: int) -> int:
    if abs(output_index) >= len(graph.output):
        raise ValueError(f"output_index {output_index} is out of range {len(graph.output)}")

    output = graph.output[output_index]
    if abs(dim_index) >= len(output.type.tensor_type.shape.dim):
        raise ValueError(f"dim_index {dim_index} is out of range {len(output.type.tensor_type.shape.dim)}")
    return output.type.tensor_type.shape.dim[dim_index].dim_value
