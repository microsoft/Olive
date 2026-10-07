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
        assert (tmp_path / "result" / f"model_{name}.onnx").is_file()
        if save_as_external_data:
            assert (tmp_path / "result" / f"model_{name}.onnx.data").is_file()

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


def test_split_vision_pooler_updates_genai_config(tmp_path):
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
    split = create_pass_from_dict(SplitVisionPooler, disable_search=True).run(
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
    assert vision["pipeline"] == [
        {
            "vision_encoder": {"filename": "model_encoder.onnx"},
            "vision_pooler_projector": {
                "filename": "model_pooler_projector.onnx",
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
            },
        }
    ]
    assert split.model_attributes[VISION_PIPELINE_KEY] == {
        "vision_encoder": "encoder",
        "vision_pooler_projector": "pooler_projector",
    }
    assert split.model_attributes["component_name_mapping"] == {
        "encoder": "vision_encoder",
        "pooler_projector": "vision_pooler_projector",
    }
    assert split.model_attributes["package_config_updates"] == [
        {
            "type": "ort_genai",
            "file_name": "genai_config.json",
            "json_paths": ["/model/vision"],
        }
    ]
    assert str(output_config_path) in split.model_attributes["additional_files"]


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
