import onnx
import pytest
from onnx import TensorProto, helper

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.dynamic_to_fixed_shape import DynamicToFixedShape
from test.utils import create_onnx_model_with_dynamic_axis


@pytest.mark.parametrize(
    ("pass_config", "err_msg"),
    [
        ({}, "dim_param and input_name cannot be both empty."),
        (
            {
                "dim_param": ["batch_size"],
                "dim_value": [1, 2],
            },
            "dim_param and dim_value must have the same number of elements.",
        ),
        (
            {
                "dim_param": ["batch_size"],
                "dim_value": [1],
                "input_name": ["input"],
                "input_shape": [[1, 3, 256, 256]],
            },
            "Cannot set both dim_param and input_name at the same time.",
        ),
        (
            {
                "dim_param": ["batch_size"],
                "dim_value": [-1],
            },
            "dim_value must be all >= 0 when dim_param is provided.",
        ),
        (
            {
                "input_name": ["input"],
                "input_shape": [[1, 0, 256, 256]],
            },
            "input_shape must be all > 0 when input_name is provided.",
        ),
    ],
)
def test_dynamic_to_fixed_shape_validator(pass_config, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        create_pass_from_dict(DynamicToFixedShape, pass_config, disable_search=True)


@pytest.mark.parametrize(
    "pass_config",
    [
        {
            "dim_param": ["batch_size"],
            "dim_value": [1],
        },
        {
            "input_name": ["input"],
            "input_shape": [[1, 1]],
        },
    ],
)
def test_dynamic_to_fixed_shape(pass_config, tmp_path):
    dynamic_shape_model_path = tmp_path / "input_model.onnx"
    create_onnx_model_with_dynamic_axis(dynamic_shape_model_path)
    input_model = ONNXModelHandler(dynamic_shape_model_path)
    input_onnx_model = input_model.load_model()
    assert input_onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch_size"
    assert input_onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch_size"

    p = create_pass_from_dict(DynamicToFixedShape, pass_config, disable_search=True)
    out = p.run(input_model, tmp_path)
    output_model = out.load_model()

    assert output_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
    assert output_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 1


def test_dynamic_to_fixed_shape_with_scalar_constant_attribute(tmp_path):
    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch_size", 1])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch_size", 1])
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["output"]),
            helper.make_node("Constant", [], ["unused"], value_int=1),
        ],
        "scalar_constant_attribute",
        [input_value],
        [output_value],
    )
    model_path = tmp_path / "input_model.onnx"
    onnx.save(helper.make_model(graph), model_path)

    p = create_pass_from_dict(
        DynamicToFixedShape,
        {"dim_param": ["batch_size"], "dim_value": [1]},
        disable_search=True,
    )
    output_model = p.run(ONNXModelHandler(model_path), tmp_path).load_model()

    assert output_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
    assert output_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 1
