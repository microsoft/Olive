# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.model import ModelConfig
from olive.passes.onnx.common import get_context_bin_file_names, get_external_data_file_names
from olive.workflows.run.builds import ComponentBuildContext
from olive.workflows.run.component_assembly import _publish_assembly, try_assemble_component_builds


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


def _write_ep_context(path: Path, location: str, binary: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("EPContext", ["input"], ["output"], ep_cache_context=location)],
        "ep-context",
        [input_info],
        [output_info],
    )
    onnx.save_model(helper.make_model(graph), path)
    if binary is not None:
        asset = path.parent / location
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(binary)


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


def _result(model_path: Path, *, additional_files=None, external_initializers=None, constant_inputs=None):
    output = _ModelOutput(model_path)
    if additional_files:
        output.olive_model_config["config"]["model_attributes"] = {"additional_files": additional_files}
    if external_initializers or constant_inputs:
        output.olive_model_config["config"] = {
            "model_path": str(model_path.parent),
            "onnx_file_name": model_path.name,
            **({"external_initializers_file_name": external_initializers} if external_initializers else {}),
            **({"constant_inputs_file_name": constant_inputs} if constant_inputs else {}),
            **({"model_attributes": {"additional_files": additional_files}} if additional_files else {}),
        }
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

    optimized_decoder = output / ".builds" / "decoder-build" / "model.onnx"
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
    assert optimized_decoder.is_file()
    assert result.get_best_candidate().model_path == str(output.resolve())
    assert (output / "embedding" / "model.onnx").stat().st_nlink == 1
    assembled_model = ModelConfig.model_validate_json(
        (output / "model_config.json").read_text(encoding="utf-8")
    ).create_model()
    assert assembled_model.model_component_names == ["decoder", "embedding"]


def test_preserves_external_data_shared_with_unbuilt_component(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    input_model = _write_shared_external_package(source)
    optimized_decoder = output / ".builds" / "decoder-build" / "model.onnx"
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


@pytest.mark.parametrize("location", ["context.bin", "nested/context.bin"])
def test_rebases_colliding_ep_context_binary_without_changing_unbuilt_component(tmp_path, location):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_ep_context(source / "decoder" / "model.onnx", location, b"source")
    _write_ep_context(source / "embedding" / "model.onnx", f"../decoder/{location}")
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_ep_context(optimized, location, b"optimized")

    try_assemble_component_builds(
        _context(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            [("decoder", ["decoder"])],
            output,
        ),
        OrderedDict([("decoder", _run_config(optimized.parent))]),
        OrderedDict([("decoder", _result(optimized))]),
    )

    rebased = get_context_bin_file_names(output / "decoder" / "model.onnx")[0]
    assert rebased != location
    assert (output / "decoder" / rebased).read_bytes() == b"optimized"
    assert (output / "decoder" / location).read_bytes() == b"source"
    assert get_context_bin_file_names(output / "embedding" / "model.onnx") == [f"../decoder/{location}"]
    assert get_context_bin_file_names(optimized) == [location]
    assert (optimized.parent / location).read_bytes() == b"optimized"


def test_distinguishes_ep_context_binaries_from_separate_builds_in_one_component_directory(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    input_model = _write_shared_external_package(source)
    decoder = output / ".builds" / "decoder" / "model.onnx"
    embedding = output / ".builds" / "embedding" / "model.onnx"
    _write_ep_context(decoder, "context.bin", b"decoder")
    _write_ep_context(embedding, "context.bin", b"embedding")

    try_assemble_component_builds(
        _context(input_model, [("decoder", ["decoder"]), ("embedding", ["embedding"])], output),
        OrderedDict([("decoder", _run_config(decoder.parent)), ("embedding", _run_config(embedding.parent))]),
        OrderedDict([("decoder", _result(decoder)), ("embedding", _result(embedding))]),
    )

    decoded = output / "shared" / "decoder.onnx"
    embedded = output / "shared" / "embedding.onnx"
    decoder_location = get_context_bin_file_names(decoded)[0]
    embedding_location = get_context_bin_file_names(embedded)[0]
    assert decoder_location != embedding_location
    assert (decoded.parent / decoder_location).read_bytes() == b"decoder"
    assert (embedded.parent / embedding_location).read_bytes() == b"embedding"


def test_keeps_openvino_xml_and_bin_together_when_context_name_collides(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_ep_context(source / "decoder" / "model.onnx", "context.xml", b"source XML")
    (source / "decoder" / "context.bin").write_bytes(b"source weights")
    _write_ep_context(source / "embedding" / "model.onnx", "../decoder/context.xml")
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_ep_context(optimized, "context.xml", b"optimized XML")
    (optimized.parent / "context.bin").write_bytes(b"optimized weights")

    try_assemble_component_builds(
        _context(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            [("decoder", ["decoder"])],
            output,
        ),
        OrderedDict([("decoder", _run_config(optimized.parent))]),
        OrderedDict([("decoder", _result(optimized))]),
    )

    reference = get_context_bin_file_names(output / "decoder" / "model.onnx")[0]
    assert reference != "context.xml"
    assert (output / "decoder" / reference).read_bytes() == b"optimized XML"
    assert (output / "decoder" / Path(reference).with_suffix(".bin")).read_bytes() == b"optimized weights"
    assert (output / "decoder" / "context.xml").read_bytes() == b"source XML"
    assert (output / "decoder" / "context.bin").read_bytes() == b"source weights"
    assert get_context_bin_file_names(output / "embedding" / "model.onnx") == ["../decoder/context.xml"]
    assert (optimized.parent / "context.bin").read_bytes() == b"optimized weights"


@pytest.mark.parametrize("existing_bin", [False, True])
def test_keeps_openvino_companion_when_only_one_file_or_neither_collides(tmp_path, existing_bin):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    if existing_bin:
        (source / "decoder" / "context.bin").write_bytes(b"other weights")
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_ep_context(optimized, "context.xml", b"optimized XML")
    (optimized.parent / "context.bin").write_bytes(b"optimized weights")

    try_assemble_component_builds(
        _context(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            [("decoder", ["decoder"])],
            output,
        ),
        OrderedDict([("decoder", _run_config(optimized.parent))]),
        OrderedDict([("decoder", _result(optimized))]),
    )

    reference = get_context_bin_file_names(output / "decoder" / "model.onnx")[0]
    assert (reference != "context.xml") == existing_bin
    assert (output / "decoder" / reference).read_bytes() == b"optimized XML"
    assert (output / "decoder" / Path(reference).with_suffix(".bin")).read_bytes() == b"optimized weights"
    if existing_bin:
        assert (output / "decoder" / "context.bin").read_bytes() == b"other weights"


def test_assembles_single_component_composite(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_onnx(source / "decoder" / "model.onnx", "source")
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_onnx(optimized, "optimized")

    assembled = try_assemble_component_builds(
        _context(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            [("decoder", ["decoder"])],
            output,
        ),
        OrderedDict([("decoder", _run_config(optimized.parent))]),
        OrderedDict([("decoder", _result(optimized))]),
    )

    assert assembled == output.resolve()
    assert onnx.load(output / "decoder" / "model.onnx").graph.name == "optimized"
    assert optimized.is_file()


def test_rejects_optimized_external_data_escaping_build_root(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_onnx(optimized, "optimized", external_data=True)
    unsafe = onnx.load(optimized, load_external_data=False)
    unsafe.graph.initializer[0].external_data[0].value = "../../other.data"
    onnx.save_model(unsafe, optimized)

    with pytest.raises(ValueError, match="escapes build output root"):
        try_assemble_component_builds(
            _context(
                ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
                [("decoder", ["decoder"])],
                output,
            ),
            OrderedDict([("decoder", _run_config(optimized.parent))]),
            OrderedDict([("decoder", _result(optimized))]),
        )
    assert not (output / "decoder" / "model.onnx").exists()


def test_optimized_package_runtime_config_replaces_source_version(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_onnx(optimized, "optimized")
    updated_config = optimized.parent / "genai_config.json"
    updated_config.write_text('{"model": "optimized"}', encoding="utf-8")

    try_assemble_component_builds(
        _context(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            [("decoder", ["decoder"])],
            output,
        ),
        OrderedDict([("decoder", _run_config(optimized.parent))]),
        OrderedDict([("decoder", _result(optimized, additional_files=[str(updated_config)]))]),
    )

    assert json.loads((output / "genai_config.json").read_text(encoding="utf-8")) == {"model": "optimized"}
    assert json.loads((source / "genai_config.json").read_text(encoding="utf-8")) == {"model": "qwen"}


def test_rejects_conflicting_package_level_updates_from_multiple_builds(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    builds = OrderedDict()
    results = OrderedDict()
    for name in ("decoder", "embedding"):
        optimized = output / ".builds" / name / "model.onnx"
        _write_onnx(optimized, name)
        updated = optimized.parent / "genai_config.json"
        updated.write_text(json.dumps({"model": name}), encoding="utf-8")
        builds[name] = _run_config(optimized.parent)
        results[name] = _result(optimized, additional_files=[str(updated)])

    with pytest.raises(ValueError, match="Conflicting component assets"):
        try_assemble_component_builds(
            _context(
                ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
                [(name, [name]) for name in builds],
                output,
            ),
            builds,
            results,
        )
    assert json.loads((source / "genai_config.json").read_text(encoding="utf-8")) == {"model": "qwen"}
    assert not (output / "decoder" / "model.onnx").exists()


def test_rejects_overwriting_unbuilt_component_shared_asset(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    embedding = source / "embedding" / "model.onnx"
    _write_onnx(embedding, "embedding", external_data=True)
    original = (embedding.parent / "weights.data").read_bytes()
    shared = source / "shared.data"
    shared.write_bytes(original)
    (embedding.parent / "weights.data").unlink()
    model = onnx.load(embedding, load_external_data=False)
    model.graph.initializer[0].external_data[0].value = "../shared.data"
    onnx.save_model(model, embedding)
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_onnx(optimized, "optimized")
    updated = optimized.parent / "shared.data"
    updated.write_bytes(b"unrelated")

    with pytest.raises(ValueError, match="conflicts with a referenced ONNX file"):
        try_assemble_component_builds(
            _context(
                ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
                [("decoder", ["decoder"])],
                output,
            ),
            OrderedDict([("decoder", _run_config(optimized.parent))]),
            OrderedDict([("decoder", _result(optimized, additional_files=[str(updated)]))]),
        )
    assert shared.read_bytes() == original
    assert not (output / "decoder" / "model.onnx").exists()


def test_optimized_directory_asset_is_preserved(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_onnx(optimized, "optimized")
    tokenizer_dir = optimized.parent / "openvino_tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "vocab.txt").write_text("tokens", encoding="utf-8")

    try_assemble_component_builds(
        _context(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            [("decoder", ["decoder"])],
            output,
        ),
        OrderedDict([("decoder", _run_config(optimized.parent))]),
        OrderedDict([("decoder", _result(optimized, additional_files=[str(tokenizer_dir)]))]),
    )

    asset = output / "decoder" / "openvino_tokenizer"
    assert (asset / "vocab.txt").read_text(encoding="utf-8") == "tokens"
    assembled = ModelConfig.model_validate_json((output / "model_config.json").read_text(encoding="utf-8"))
    assert str(asset) in assembled.config["model_components"][0]["config"]["model_attributes"]["additional_files"]


@pytest.mark.parametrize(
    ("field", "argument"),
    [
        ("external_initializers_file_name", "external_initializers"),
        ("constant_inputs_file_name", "constant_inputs"),
    ],
)
def test_nested_initializers_keep_rebased_name_without_overwriting_source_asset(tmp_path, field, argument):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    (source / "decoder" / "params").mkdir()
    original = source / "decoder" / "params" / "initializers.npz"
    original.write_bytes(b"original")
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_onnx(optimized, "optimized")
    (optimized.parent / "params").mkdir()
    (optimized.parent / "params" / "initializers.npz").write_bytes(b"optimized")

    try_assemble_component_builds(
        _context(
            ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
            [("decoder", ["decoder"])],
            output,
        ),
        OrderedDict([("decoder", _run_config(optimized.parent))]),
        OrderedDict([("decoder", _result(optimized, **{argument: "params/initializers.npz"}))]),
    )

    assembled = ModelConfig.model_validate_json((output / "model_config.json").read_text(encoding="utf-8"))
    optimized_component = assembled.config["model_components"][0]["config"]
    assert (output / "decoder" / "params" / "initializers.npz").read_bytes() == b"original"
    assert (output / "decoder" / optimized_component[field]).read_bytes() == b"optimized"


def test_publisher_rejects_existing_symlink_without_following_it(tmp_path):
    temporary = tmp_path / "staged"
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    (temporary / "decoder").mkdir(parents=True)
    (temporary / "decoder" / "model.onnx").write_bytes(b"new")
    output.mkdir()
    outside.mkdir()
    try:
        (output / "decoder").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Cannot create directory symlinks: {exc}")

    with pytest.raises(ValueError, match="symlink or junction"):
        _publish_assembly(temporary, output)
    assert not list(outside.iterdir())


def test_rejects_optimized_artifact_symlinks(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_package(source)
    optimized = output / ".builds" / "decoder" / "model.onnx"
    _write_onnx(optimized, "optimized", external_data=True)
    outside = tmp_path / "outside.data"
    outside.write_bytes(b"not an ONNX asset")
    (optimized.parent / "weights.data").unlink()
    try:
        (optimized.parent / "weights.data").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Cannot create file symlinks: {exc}")

    with pytest.raises(ValueError, match="symlink or junction"):
        try_assemble_component_builds(
            _context(
                ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(source)}}),
                [("decoder", ["decoder"])],
                output,
            ),
            OrderedDict([("decoder", _run_config(optimized.parent))]),
            OrderedDict([("decoder", _result(optimized))]),
        )
    assert outside.read_bytes() == b"not an ONNX asset"
    assert not (output / "decoder" / "model.onnx").exists()


def test_publisher_rolls_back_partial_publication_without_removing_unrelated_files(tmp_path, monkeypatch):
    temporary = tmp_path / "staged"
    output = tmp_path / "output"
    (temporary / "decoder").mkdir(parents=True)
    (temporary / "embedding").mkdir()
    (temporary / "decoder" / "model.onnx").write_bytes(b"decoder")
    (temporary / "embedding" / "model.onnx").write_bytes(b"embedding")
    output.mkdir()
    (output / "custom.txt").write_text("untouched", encoding="utf-8")
    original_link = os.link

    def fail_second_publication(path, destination):
        if path.name == "model.onnx" and path.parent.name == "embedding":
            raise OSError("publish failed")
        return original_link(path, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "link", fail_second_publication)
        with pytest.raises(OSError, match="publish failed"):
            _publish_assembly(temporary, output)
    assert (output / "custom.txt").read_text(encoding="utf-8") == "untouched"
    assert not (output / "decoder").exists()
    assert not (output / "embedding").exists()


def test_publisher_does_not_overwrite_file_created_during_publication(tmp_path, monkeypatch):
    temporary = tmp_path / "staged"
    output = tmp_path / "output"
    (temporary / "decoder").mkdir(parents=True)
    (temporary / "embedding").mkdir()
    (temporary / "decoder" / "model.onnx").write_bytes(b"decoder")
    (temporary / "embedding" / "model.onnx").write_bytes(b"embedding")
    original_link = os.link

    def concurrent_file(path, destination):
        if path.parent.name == "embedding":
            destination.write_bytes(b"from another process")
        return original_link(path, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "link", concurrent_file)
        with pytest.raises(FileExistsError):
            _publish_assembly(temporary, output)
    assert (output / "embedding" / "model.onnx").read_bytes() == b"from another process"
    assert not (output / "decoder").exists()
