# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from olive.common.ort_genai_config import ORT_GENAI_CONFIG_UPDATES_KEY
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import VISION_PIPELINE_KEY
from olive.passes.onnx.split_vision_pooler import SplitVisionPooler


def _make_model(path, pooler_name="vision/pooler/Add"):
    def value_info(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["pixel_position_ids"], ["shared"], name="vision/shared"),
            helper.make_node("Add", ["pixel_values", "shared"], ["encoded"], name="vision/encoder/Add"),
            helper.make_node("QuantizeLinear", ["encoded", "scale", "zero_point"], ["quantized"], name="vision/Q"),
            helper.make_node("DequantizeLinear", ["quantized", "scale", "zero_point"], ["features"], name="vision/DQ"),
            helper.make_node("Add", ["features", "shared"], ["pooled"], name=pooler_name),
            helper.make_node("Add", ["pooled", "bias"], ["image_features"], name="vision/projector/Add"),
        ],
        "vision",
        [value_info("pixel_values"), value_info("pixel_position_ids")],
        [value_info("image_features")],
        [
            helper.make_tensor("scale", TensorProto.FLOAT, [], [0.25]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [], [0]),
            helper.make_tensor("bias", TensorProto.FLOAT, [1, 2], [1, 2]),
        ],
        value_info=[value_info("encoded")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=10), path)


def _make_quantized_boundary_model(path, zero_point_type):
    def value_info(name, elem_type=TensorProto.FLOAT):
        return helper.make_tensor_value_info(name, elem_type, [1, 2])

    quantize_inputs = ["encoded", "scale"]
    dequantize_inputs = ["quantized", "scale"]
    initializers = [helper.make_tensor("scale", TensorProto.FLOAT, [], [0.25])]
    if zero_point_type is not None:
        quantize_inputs.append("zero_point")
        dequantize_inputs.append("zero_point")
        initializers.append(helper.make_tensor("zero_point", zero_point_type, [], [0]))

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["pixel_position_ids"], ["shared"], name="vision/shared"),
            helper.make_node("Add", ["pixel_values", "shared"], ["encoded"], name="vision/encoder/Add"),
            helper.make_node("QuantizeLinear", quantize_inputs, ["quantized"], name="vision/Q"),
            helper.make_node("DequantizeLinear", dequantize_inputs, ["features"], name="vision/pooler/DQ"),
            helper.make_node("Add", ["features", "shared"], ["image_features"], name="vision/projector/Add"),
        ],
        "vision",
        [value_info("pixel_values"), value_info("pixel_position_ids")],
        [value_info("image_features")],
        initializers,
        value_info=[value_info("encoded")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=10), path)


def _make_subgraph_initializer_model(path):
    def value_info(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])

    branch_output = value_info("conditional_bias")
    true_branch = helper.make_graph(
        [helper.make_node("Identity", ["branch_bias"], ["conditional_bias"])],
        "true_branch",
        [],
        [branch_output],
    )
    false_branch = helper.make_graph(
        [helper.make_node("Identity", ["branch_bias"], ["conditional_bias"])],
        "false_branch",
        [],
        [value_info("conditional_bias")],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["pixel_position_ids"], ["shared"], name="vision/shared"),
            helper.make_node("Add", ["pixel_values", "shared"], ["encoded"], name="vision/encoder/Add"),
            helper.make_node("Add", ["encoded", "shared"], ["pooled"], name="vision/pooler/Add"),
            helper.make_node(
                "If",
                ["condition"],
                ["conditional_bias"],
                name="vision/projector/If",
                then_branch=true_branch,
                else_branch=false_branch,
            ),
            helper.make_node("Add", ["pooled", "conditional_bias"], ["image_features"], name="vision/projector/Add"),
        ],
        "vision",
        [value_info("pixel_values"), value_info("pixel_position_ids")],
        [value_info("image_features")],
        [
            helper.make_tensor("condition", TensorProto.BOOL, [], [True]),
            helper.make_tensor("branch_bias", TensorProto.FLOAT, [1, 2], [1, 2]),
        ],
        value_info=[value_info("encoded")],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=10), path)


@pytest.mark.parametrize("save_as_external_data", [False, True])
def test_split_vision_pooler_preserves_outputs_and_qdq_metadata(tmp_path, save_as_external_data):
    input_path = tmp_path / "input.onnx"
    _make_model(input_path)
    split = create_pass_from_dict(
        SplitVisionPooler, {"save_as_external_data": save_as_external_data}, disable_search=True
    ).run(ONNXModelHandler(input_path), str(tmp_path / "result"))

    assert isinstance(split, CompositeModelHandler)
    assert split.model_component_names == ["encoder", "pooler_projector"]
    encoder, pooler = split.model_components
    assert encoder.io_config["input_names"] == ["pixel_values", "pixel_position_ids"]
    assert encoder.io_config["output_names"] == ["vision_features"]
    assert pooler.io_config["input_names"] == ["pixel_position_ids", "vision_features"]
    assert pooler.io_config["output_names"] == ["image_features"]
    for component in (encoder, pooler):
        model = onnx.load(component.model_path)
        assert "vision/shared" in {node.name for node in model.graph.node}
        boundary_value = next(
            value for value in (*model.graph.input, *model.graph.output) if value.name == "vision_features"
        )
        assert boundary_value.type.tensor_type.elem_type == TensorProto.FLOAT
        assert [dim.dim_value for dim in boundary_value.type.tensor_type.shape.dim] == [1, 2]
    for component, name in zip((encoder, pooler), split.model_component_names):
        assert isinstance(component, ONNXModelHandler)
        metadata = {entry.key: entry.value for entry in onnx.load(component.model_path).metadata_props}
        assert metadata["gemma_vision_split_component"] == name
        assert metadata["gemma_vision_split_boundary"] == "vision_features"
        assert (tmp_path / "result" / f"{name}.onnx").is_file()
        if save_as_external_data:
            assert (tmp_path / "result" / f"{name}.onnx.data").is_file()

    inputs = {
        "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
        "pixel_position_ids": np.array([[0.25, 0.5]], dtype=np.float32),
    }
    expected = ort.InferenceSession(str(input_path)).run(None, inputs)[0]
    features = ort.InferenceSession(str(encoder.model_path)).run(None, inputs)[0]
    actual = ort.InferenceSession(str(pooler.model_path)).run(
        None, {"pixel_position_ids": inputs["pixel_position_ids"], "vision_features": features}
    )[0]
    np.testing.assert_array_equal(actual, expected)


def test_split_vision_pooler_uses_unique_explicit_external_data_names(tmp_path):
    input_path = tmp_path / "input.onnx"
    _make_model(input_path)
    split = create_pass_from_dict(
        SplitVisionPooler,
        {"save_as_external_data": True, "external_data_name": "weights.bin", "size_threshold": 0},
        disable_search=True,
    ).run(ONNXModelHandler(input_path), str(tmp_path / "result"))

    for component, name in zip(split.model_components, split.model_component_names):
        external_data_path = tmp_path / "result" / f"{name}.weights.bin"
        assert external_data_path.is_file()
        model = onnx.load(component.model_path, load_external_data=True)
        onnx.checker.check_model(model)

    inputs = {
        "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
        "pixel_position_ids": np.array([[0.25, 0.5]], dtype=np.float32),
    }
    expected = ort.InferenceSession(str(input_path)).run(None, inputs)[0]
    encoder, pooler = split.model_components
    features = ort.InferenceSession(str(encoder.model_path)).run(None, inputs)[0]
    actual = ort.InferenceSession(str(pooler.model_path)).run(
        None, {"pixel_position_ids": inputs["pixel_position_ids"], "vision_features": features}
    )[0]
    np.testing.assert_array_equal(actual, expected)


def test_split_vision_pooler_embeds_external_source_data_by_default(tmp_path):
    input_path = tmp_path / "input.onnx"
    _make_model(input_path)
    model = onnx.load(input_path)
    for initializer in model.graph.initializer:
        initializer.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(initializer), initializer.name))
    onnx.save_model(
        model,
        input_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="input.data",
        size_threshold=0,
    )
    assert (tmp_path / "input.data").is_file()

    split = create_pass_from_dict(SplitVisionPooler, disable_search=True).run(
        ONNXModelHandler(input_path), str(tmp_path / "result")
    )

    for component in split.model_components:
        model = onnx.load(component.model_path, load_external_data=False)
        assert all(initializer.data_location == TensorProto.DEFAULT for initializer in model.graph.initializer)
        onnx.load(component.model_path, load_external_data=True)


@pytest.mark.parametrize(
    ("zero_point_type", "expected_type"),
    [(None, TensorProto.UINT8), (TensorProto.UINT8, TensorProto.UINT8), (TensorProto.INT8, TensorProto.INT8)],
)
def test_split_vision_pooler_infers_quantize_linear_boundary_type(tmp_path, zero_point_type, expected_type):
    input_path = tmp_path / "input.onnx"
    _make_quantized_boundary_model(input_path, zero_point_type)
    split = create_pass_from_dict(SplitVisionPooler, disable_search=True).run(
        ONNXModelHandler(input_path), str(tmp_path / "result")
    )

    for component in split.model_components:
        model = onnx.load(component.model_path)
        boundary = next(value for value in (*model.graph.input, *model.graph.output) if value.name == "vision_features")
        assert boundary.type.tensor_type.elem_type == expected_type


def test_split_vision_pooler_preserves_initializer_used_by_nested_subgraph(tmp_path):
    input_path = tmp_path / "input.onnx"
    _make_subgraph_initializer_model(input_path)
    split = create_pass_from_dict(SplitVisionPooler, disable_search=True).run(
        ONNXModelHandler(input_path), str(tmp_path / "result")
    )
    encoder, pooler = split.model_components

    pooler_model = onnx.load(pooler.model_path)
    assert "branch_bias" in {initializer.name for initializer in pooler_model.graph.initializer}

    inputs = {
        "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
        "pixel_position_ids": np.array([[0.25, 0.5]], dtype=np.float32),
    }
    expected = ort.InferenceSession(str(input_path)).run(None, inputs)[0]
    features = ort.InferenceSession(str(encoder.model_path)).run(None, inputs)[0]
    actual = ort.InferenceSession(str(pooler.model_path)).run(
        None, {"pixel_position_ids": inputs["pixel_position_ids"], "vision_features": features}
    )[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "stage_session_options",
    [
        None,
        {
            "vision_encoder": {"provider_options": [{"OpenVINO": {"device_type": "NPU"}}]},
            "vision_pooler_projector": {"provider_options": []},
        },
        {
            "vision_pooler_projector": {"provider_options": [{"OpenVINO": {"device_type": "CPU"}}]},
        },
    ],
)
def test_split_vision_pooler_updates_genai_config(tmp_path, stage_session_options):
    input_path = tmp_path / "input.onnx"
    _make_model(input_path)
    genai_config_path = tmp_path / "genai_config.json"
    genai_config_path.write_text(
        json.dumps(
            {
                "model": {
                    "type": "gemma4",
                    "vision": {
                        "filename": "vision_encoder/model.onnx",
                        "inputs": {"pixel_values": "pixel_values"},
                        "outputs": {"image_features": "image_features"},
                        "session_options": {
                            "log_id": "onnxruntime-genai",
                            "provider_options": [],
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    split = create_pass_from_dict(
        SplitVisionPooler, {"stage_session_options": stage_session_options}, disable_search=True
    ).run(
        ONNXModelHandler(
            input_path,
            model_attributes={"additional_files": [str(genai_config_path)]},
        ),
        str(tmp_path / "result"),
    )

    output_config_path = tmp_path / "result" / "genai_config.json"
    vision = json.loads(output_config_path.read_text(encoding="utf-8"))["model"]["vision"]
    assert "filename" not in vision
    assert vision["inputs"] == {"pixel_values": "pixel_values"}
    expected_encoder = {"filename": "encoder.onnx"}
    if stage_session_options is not None and "vision_encoder" in stage_session_options:
        expected_encoder["session_options"] = stage_session_options["vision_encoder"]
    expected_pooler_options = (
        stage_session_options["vision_pooler_projector"]
        if stage_session_options is not None and "vision_pooler_projector" in stage_session_options
        else {"log_id": "onnxruntime-genai", "provider_options": []}
    )
    assert vision["pipeline"] == [
        {
            "vision_encoder": expected_encoder,
            "vision_pooler_projector": {
                "filename": "pooler_projector.onnx",
                "session_options": expected_pooler_options,
            },
        }
    ]
    assert json.loads(genai_config_path.read_text(encoding="utf-8"))["model"]["vision"]["session_options"] == {
        "log_id": "onnxruntime-genai",
        "provider_options": [],
    }
    assert split.model_attributes[VISION_PIPELINE_KEY] == {
        "vision_encoder": "encoder",
        "vision_pooler_projector": "pooler_projector",
    }
    assert split.model_attributes[ORT_GENAI_CONFIG_UPDATES_KEY] == [
        {"file_name": "genai_config.json", "json_paths": ["/model/vision"]}
    ]
    assert "component_name_mapping" not in split.model_attributes
    assert "package_config_updates" not in split.model_attributes
    assert str(output_config_path) in split.model_attributes["additional_files"]


def test_split_vision_pooler_rejects_unknown_session_option_stage(tmp_path):
    input_path = tmp_path / "input.onnx"
    _make_model(input_path)
    genai_config_path = tmp_path / "genai_config.json"
    genai_config_path.write_text(
        json.dumps({"model": {"vision": {"filename": "vision_encoder/model.onnx"}}}),
        encoding="utf-8",
    )
    split = create_pass_from_dict(
        SplitVisionPooler,
        {"stage_session_options": {"wrong_stage": {"provider_options": []}}},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="Unknown vision pipeline session option stage"):
        split.run(
            ONNXModelHandler(input_path, model_attributes={"additional_files": [str(genai_config_path)]}),
            str(tmp_path / "result"),
        )


@pytest.mark.parametrize(
    ("pooler_name", "primary_input", "message"),
    [
        ("vision/head/Add", "pixel_values", "no nodes contain pooler marker"),
        ("vision/pooler/Add", "missing_input", "expected one boundary value"),
    ],
)
def test_split_vision_pooler_rejects_missing_boundary(tmp_path, pooler_name, primary_input, message):
    input_path = tmp_path / "input.onnx"
    _make_model(input_path, pooler_name)
    split_pass = create_pass_from_dict(SplitVisionPooler, {"primary_input": primary_input}, disable_search=True)
    with pytest.raises(ValueError, match=message):
        split_pass.run(ONNXModelHandler(input_path), str(tmp_path / "result"))


def test_split_vision_pooler_rejects_multiple_boundaries(tmp_path):
    input_path = tmp_path / "input.onnx"
    _make_model(input_path)
    model = onnx.load(input_path)
    model.graph.node[4].input[1] = "encoded"
    onnx.save(model, input_path)
    split_pass = create_pass_from_dict(SplitVisionPooler, disable_search=True)
    with pytest.raises(ValueError, match="expected one boundary value"):
        split_pass.run(ONNXModelHandler(input_path), str(tmp_path / "result"))
