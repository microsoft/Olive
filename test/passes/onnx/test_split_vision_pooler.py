# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from olive.cache import OliveCache
from olive.hardware import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
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


def _make_mobius_package(source):
    source.mkdir()
    vision = source / "vision_encoder"
    vision.mkdir()
    _make_model(vision / "model.onnx")
    for name in ("decoder", "embedding", "audio_encoder"):
        folder = source / name
        folder.mkdir()
        shutil.copy2(vision / "model.onnx", folder / "model.onnx")

    genai = {
        "model": {
            "type": "gemma4",
            "decoder": {"filename": "decoder/model.onnx", "session_options": {"provider_options": []}},
            "vision": {
                "filename": "vision_encoder/model.onnx",
                "spatial_merge_size": 2,
                "session_options": {"provider_options": [], "log_id": "original"},
                "pipeline": {
                    "encoder": {"filename": "vision_encoder/model_encoder.onnx"},
                    "projector": {"filename": "vision_encoder/model_pooler_projector.onnx"},
                },
            },
        },
        "search": {"top_k": 1},
    }
    (source / "genai_config.json").write_text(json.dumps(genai))
    (source / "image_processor.json").write_text('{"preserve": true}')
    names = ["decoder", "vision_encoder", "audio_encoder", "embedding"]
    package = CompositeModelHandler(
        [ONNXModelHandler(source / name, onnx_file_name="model.onnx") for name in names],
        names,
        model_path=source,
        model_attributes={
            "no_flatten": True,
            "additional_files": [str(source / "genai_config.json"), str(source / "image_processor.json")],
        },
    )
    (source / "model_config.json").write_text(json.dumps(package.to_json()))


@pytest.mark.parametrize(
    ("execution_provider", "provider_name"),
    [
        ("QNNExecutionProvider", "QNN"),
        ("OpenVINOExecutionProvider", "OpenVINO"),
        ("VitisAIExecutionProvider", "VitisAI"),
    ],
)
def test_split_vision_pooler_packages_mobius_export_without_modifying_source(
    tmp_path, execution_provider, provider_name
):
    assert not SplitVisionPooler.is_accelerator_agnostic(
        AcceleratorSpec(accelerator_type="npu", execution_provider=execution_provider)
    )
    source = tmp_path / "export"
    _make_mobius_package(source)
    original_genai = (source / "genai_config.json").read_bytes()
    original_metadata = (source / "model_config.json").read_bytes()
    input_path = tmp_path / "quantized.onnx"
    _make_model(input_path)
    options = {"htp_performance_mode": "burst"} if provider_name == "QNN" else {"device_type": "NPU"}
    split_pass = create_pass_from_dict(
        SplitVisionPooler,
        {"source_package_dir": str(source), "provider_options": options, "save_as_external_data": True},
        accelerator_spec=AcceleratorSpec(accelerator_type="npu", execution_provider=execution_provider),
        disable_search=True,
    )
    output = tmp_path / "final"
    result = split_pass.run(ONNXModelHandler(input_path), str(output))

    assert result.model_component_names == ["decoder", "vision_encoder", "audio_encoder", "embedding"]
    assert result.model_attributes["no_flatten"]
    for name in result.model_component_names:
        assert (output / name / "model.onnx").is_file()
    assert (output / "image_processor.json").read_text() == '{"preserve": true}'
    assert (source / "genai_config.json").read_bytes() == original_genai
    assert (source / "model_config.json").read_bytes() == original_metadata
    assert not (source / "vision_encoder" / "model_encoder.onnx").exists()

    vision = output / "vision_encoder"
    for name in ("encoder", "pooler_projector"):
        model = vision / f"model_{name}.onnx"
        onnx.checker.check_model(str(model))
        assert (vision / f"model_{name}.onnx.data").is_file()
    assert json.loads((output / "genai_config.json").read_text()) == {
        **json.loads(original_genai),
        "model": {
            **json.loads(original_genai)["model"],
            "vision": {
                **json.loads(original_genai)["model"]["vision"],
                "session_options": {
                    "log_id": f"onnxruntime-genai-{provider_name.lower()}",
                    "provider_options": [{provider_name: options}],
                },
                "pipeline": {
                    "encoder": {
                        "filename": "vision_encoder/model_encoder.onnx",
                        "inputs": ["pixel_values", "pixel_position_ids"],
                        "outputs": ["vision_features"],
                    },
                    "projector": {
                        "filename": "vision_encoder/model_pooler_projector.onnx",
                        "inputs": ["vision_features", "pixel_position_ids"],
                        "outputs": ["image_features"],
                        "run_on_cpu": True,
                    },
                },
            },
        },
    }
    metadata = json.loads((output / "model_config.json").read_text())
    assert metadata["config"]["model_path"] == str(output)
    assert metadata["config"]["model_components"][1]["config"]["model_path"] == str(vision)
    assert str(output / "genai_config.json") in metadata["config"]["model_attributes"]["additional_files"]

    cache = OliveCache({"cache_dir": tmp_path / "cache"})
    cache.cache_model("vision", result.to_json())
    saved = tmp_path / "saved"
    cache.save_model("vision", output_dir=saved, overwrite=True)
    assert (saved / "vision_encoder" / "model_encoder.onnx.data").is_file()
    assert (saved / "image_processor.json").read_text() == '{"preserve": true}'
    assert json.loads((saved / "genai_config.json").read_text()) == json.loads(
        (output / "genai_config.json").read_text()
    )
    saved_metadata = json.loads((saved / "model_config.json").read_text())
    assert saved_metadata["config"]["model_path"] == str(saved)
    assert saved_metadata["config"]["model_components"][1]["config"]["model_path"] == str(saved / "vision_encoder")


def test_split_vision_pooler_rejects_unsupported_package_provider(tmp_path):
    source = tmp_path / "export"
    _make_mobius_package(source)
    input_path = tmp_path / "input.onnx"
    _make_model(input_path)
    split_pass = create_pass_from_dict(SplitVisionPooler, {"source_package_dir": str(source)}, disable_search=True)
    with pytest.raises(ValueError, match="Unsupported vision pipeline provider"):
        split_pass.run(ONNXModelHandler(input_path), str(tmp_path / "final"))
