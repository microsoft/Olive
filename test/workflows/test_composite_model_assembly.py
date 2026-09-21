# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.model import ModelConfig
from olive.workflows.run.composite_model_assembly import try_assemble_composite_model_builds


def _write_onnx(path: Path, graph_name: str, external_data: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.ones((1,), dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "bias"], ["output"])],
        graph_name,
        [input_info],
        [output_info],
        [initializer],
    )
    model = helper.make_model(graph)
    if external_data:
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.data",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, path)


def _write_package(path: Path) -> None:
    for component in ("decoder", "embedding", "vision_encoder"):
        _write_onnx(path / component / "model.onnx", f"source-{component}")
    (path / "vision_encoder" / "processor.json").write_text("{}", encoding="utf-8")
    (path / "genai_config.json").write_text('{"model": "qwen"}', encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model_config.json").write_text('{"stale": true}', encoding="utf-8")


class _ModelOutput:
    def __init__(self, model_path: Path):
        self.olive_model_config = {
            "type": "onnxmodel",
            "config": {
                "model_path": str(model_path),
            },
        }
        self.model_path = str(model_path)

    def _update_with_model_config(self, model_config):
        self.olive_model_config = model_config
        self.model_path = model_config["config"]["model_path"]


def _result(model_path: Path):
    output = _ModelOutput(model_path)
    return SimpleNamespace(get_best_candidate=lambda: output), output


def _run_config(output_dir: Path):
    return SimpleNamespace(engine=SimpleNamespace(output_dir=output_dir))


def test_assembles_optimized_and_unbuilt_composite_components(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)

    optimized_decoder = output / "decoder-build" / "model.onnx"
    _write_onnx(optimized_decoder, "optimized-decoder", external_data=True)
    assert (optimized_decoder.parent / "weights.data").is_file()
    (optimized_decoder.parent / "footprint.json").write_text("{}", encoding="utf-8")
    result, model_output = _result(optimized_decoder)

    assembled = try_assemble_composite_model_builds(
        ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
        OrderedDict([("decoder-build", ["decoder"])]),
        OrderedDict([("decoder-build", _run_config(optimized_decoder.parent))]),
        OrderedDict([("decoder-build", result)]),
        output,
    )

    assert assembled == output.resolve()
    assert onnx.load(output / "decoder" / "model.onnx").graph.name == "optimized-decoder"
    assert (output / "decoder" / "model.onnx.data").is_file()
    assert onnx.load(output / "embedding" / "model.onnx").graph.name == "source-embedding"
    assert onnx.load(output / "vision_encoder" / "model.onnx").graph.name == "source-vision_encoder"
    assert (output / "vision_encoder" / "processor.json").is_file()
    assert (output / "genai_config.json").read_text(encoding="utf-8") == '{"model": "qwen"}'
    assert (output / "tokenizer.json").is_file()
    assert not (output / "decoder-build").exists()
    assert (source / "model_config.json").read_text(encoding="utf-8") == '{"stale": true}'
    assert onnx.load(source / "decoder" / "model.onnx").graph.name == "source-decoder"

    model_config = json.loads((output / "model_config.json").read_text(encoding="utf-8"))
    assert model_config["type"] == "compositemodel"
    assert model_config["config"]["model_component_names"] == ["decoder", "embedding", "vision_encoder"]
    assert model_config["config"]["model_attributes"]["assembled_components"] == ["decoder"]
    assert set(model_config["config"]["model_attributes"]["additional_files"]) == {
        str(output / "genai_config.json"),
        str(output / "tokenizer.json"),
    }
    for component in model_config["config"]["model_components"]:
        assert Path(component["config"]["model_path"]).is_relative_to(output)
        assert component["config"]["onnx_file_name"] == "model.onnx"
    assembled_model = ModelConfig.model_validate(model_config).create_model()
    assert assembled_model.model_component_names == ["decoder", "embedding", "vision_encoder"]

    assert model_output.model_path == str(output)
    assert model_output.olive_model_config == model_config


def test_assembles_into_existing_default_engine_output(tmp_path):
    output = tmp_path / "work"
    source = output / "exported"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    _write_package(source)

    optimized_decoder = output / "output" / "decoder-build" / "model.onnx"
    _write_onnx(optimized_decoder, "optimized-decoder")
    result, _ = _result(optimized_decoder)

    assembled = try_assemble_composite_model_builds(
        ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
        OrderedDict([("decoder-build", ["decoder"])]),
        OrderedDict([("decoder-build", _run_config(optimized_decoder.parent))]),
        OrderedDict([("decoder-build", result)]),
        output,
    )

    assert assembled == output.resolve()
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert onnx.load(output / "decoder" / "model.onnx").graph.name == "optimized-decoder"
    assert onnx.load(output / "embedding" / "model.onnx").graph.name == "source-embedding"
    assert onnx.load(source / "decoder" / "model.onnx").graph.name == "source-decoder"
    assert not (output / "output").exists()


def test_does_not_assemble_without_workflow_output(tmp_path):
    source = tmp_path / "source"
    _write_package(source)
    optimized_decoder = tmp_path / "decoder-build" / "model.onnx"
    _write_onnx(optimized_decoder, "optimized-decoder")
    result, _ = _result(optimized_decoder)

    assert (
        try_assemble_composite_model_builds(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            OrderedDict([("decoder-build", ["decoder"])]),
            OrderedDict([("decoder-build", _run_config(optimized_decoder.parent))]),
            OrderedDict([("decoder-build", result)]),
            None,
        )
        is None
    )


def test_does_not_assemble_whole_model_build(tmp_path):
    source = tmp_path / "source"
    _write_package(source)

    assert (
        try_assemble_composite_model_builds(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            OrderedDict([("whole-model", [])]),
            OrderedDict(),
            OrderedDict(),
            tmp_path / "output",
        )
        is None
    )
    assert not (tmp_path / "output").exists()


def test_rejects_overlapping_component_builds(tmp_path):
    source = tmp_path / "source"
    _write_package(source)
    first_model = tmp_path / "first" / "model.onnx"
    second_model = tmp_path / "second" / "model.onnx"
    _write_onnx(first_model, "first")
    _write_onnx(second_model, "second")
    first_result, _ = _result(first_model)
    second_result, _ = _result(second_model)

    with pytest.raises(ValueError, match="overlapping components"):
        try_assemble_composite_model_builds(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            OrderedDict(
                [
                    ("first", ["decoder"]),
                    ("second", ["decoder"]),
                ]
            ),
            OrderedDict(
                [
                    ("first", _run_config(first_model.parent)),
                    ("second", _run_config(second_model.parent)),
                ]
            ),
            OrderedDict(
                [
                    ("first", first_result),
                    ("second", second_result),
                ]
            ),
            tmp_path / "output",
        )
