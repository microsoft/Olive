# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.model import ModelConfig
from olive.passes.onnx.common import get_external_data_file_names
from olive.workflows.run.builds import ComponentBuildContext
from olive.workflows.run.component_assembly import try_assemble_component_builds


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
    for component in ("decoder", "embedding"):
        _write_onnx(path / component / "model.onnx", f"source-{component}")
    (path / "genai_config.json").write_text('{"model": "qwen"}', encoding="utf-8")


def _write_shared_external_package(path: Path) -> ModelConfig:
    shared = path / "shared"
    decoder_path = shared / "decoder.onnx"
    _write_onnx(decoder_path, "source-decoder", external_data=True)
    embedding_model = onnx.load(decoder_path, load_external_data=False)
    embedding_model.graph.name = "source-embedding"
    embedding_path = shared / "embedding.onnx"
    onnx.save_model(embedding_model, embedding_path)
    return ModelConfig.model_validate(
        {
            "type": "CompositeModel",
            "config": {
                "model_path": str(path),
                "model_component_names": ["decoder", "embedding"],
                "model_components": [
                    {
                        "type": "ONNXModel",
                        "config": {"model_path": str(shared), "onnx_file_name": "decoder.onnx"},
                    },
                    {
                        "type": "ONNXModel",
                        "config": {"model_path": str(shared), "onnx_file_name": "embedding.onnx"},
                    },
                ],
            },
        }
    )


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
    return SimpleNamespace(get_best_candidate=lambda: output)


def _run_config(output_dir: Path):
    return SimpleNamespace(engine=SimpleNamespace(output_dir=output_dir))


def _context(input_model: ModelConfig, components, output_dir: Path) -> ComponentBuildContext:
    return ComponentBuildContext(input_model, OrderedDict(components), output_dir)


def test_dispatches_hf_component_assembly(monkeypatch, tmp_path):
    expected = tmp_path / "assembled"
    context = _context(
        ModelConfig.model_validate({"type": "HfModel", "model_path": "org/model"}),
        [("decoder", ["decoder"])],
        expected,
    )

    def fake_assemble(build_configs, results, output_dir):
        assert build_configs == {"decoder": "config"}
        assert results == OrderedDict([("decoder", "result")])
        assert output_dir == expected
        return expected

    hf_module = pytest.importorskip("olive.workflows.run.hf_component_assembly")
    monkeypatch.setattr(hf_module, "try_assemble_hf_component_builds", fake_assemble)

    assert (
        try_assemble_component_builds(
            context,
            {"decoder": "config"},
            OrderedDict([("decoder", "result")]),
        )
        == expected
    )


def test_assembles_optimized_and_unbuilt_composite_components(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    (output / "decoder").mkdir(parents=True)
    (output / "decoder" / "custom.txt").write_text("custom", encoding="utf-8")

    optimized_decoder = output / "decoder-build" / "model.onnx"
    _write_onnx(optimized_decoder, "optimized-decoder")
    result = _result(optimized_decoder)

    assembled = try_assemble_component_builds(
        _context(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            [("decoder-build", ["decoder"])],
            output,
        ),
        OrderedDict([("decoder-build", _run_config(optimized_decoder.parent))]),
        OrderedDict([("decoder-build", result)]),
    )

    assert assembled == output.resolve()
    assert onnx.load(output / "decoder" / "model.onnx").graph.name == "optimized-decoder"
    assert onnx.load(output / "embedding" / "model.onnx").graph.name == "source-embedding"
    assert (output / "genai_config.json").is_file()
    assert (output / "decoder" / "custom.txt").is_file()
    assert not (output / "decoder-build").exists()
    assert (output / "embedding" / "model.onnx").stat().st_nlink == 1
    assembled_model = ModelConfig.model_validate_json(
        (output / "model_config.json").read_text(encoding="utf-8")
    ).create_model()
    assert assembled_model.model_component_names == ["decoder", "embedding"]


def test_preserves_external_data_shared_with_unbuilt_component(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    input_model = _write_shared_external_package(source)
    optimized_decoder = output / "decoder-build" / "model.onnx"
    _write_onnx(optimized_decoder, "optimized-decoder", external_data=True)
    result = _result(optimized_decoder)

    try_assemble_component_builds(
        _context(input_model, [("decoder-build", ["decoder"])], output),
        OrderedDict([("decoder-build", _run_config(optimized_decoder.parent))]),
        OrderedDict([("decoder-build", result)]),
    )

    assert (output / "shared" / "weights.data").is_file()
    assert (output / "shared" / "weights.data").stat().st_nlink == 1
    assert onnx.load(output / "shared" / "embedding.onnx").graph.name == "source-embedding"
    assert onnx.load(output / "shared" / "decoder.onnx").graph.name == "optimized-decoder"
    assert get_external_data_file_names(output / "shared" / "decoder.onnx") != ["weights.data"]
