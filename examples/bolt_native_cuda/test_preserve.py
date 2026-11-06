# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Network-free preservation tests; opaque packed nodes are not numerical fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import onnx_ir as ir
import pytest

from examples.bolt_native_cuda import preserve as recipe
from olive.hardware.accelerator import Device
from olive.model.config.model_config import ModelConfig
from olive.model.handler.onnx import ONNXModelHandler
from olive.workflows.run.config import RunConfig


@pytest.fixture(name="package")
def native_package_fixture(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    loading = {
        "revision": "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a",
        "native_iq2_storage": True,
        "full_checkpoint_dequantization": False,
    }
    for name in recipe.COMPONENTS:
        folder = root / name
        folder.mkdir()
        (folder / "model.onnx.data").write_bytes(b"header!!" + bytes(range(64)) + b"tail")
        tensor = ir.ExternalTensor(
            "model.onnx.data", 8, 64, ir.DataType.UINT8, shape=ir.Shape([64]), name="packed", base_dir=folder
        )
        packed = ir.val("packed", const_value=tensor)
        x = ir.val("x", ir.DataType.FLOAT, [1, 1])
        nodes = []
        value = x
        for op in sorted(recipe.PACKED_OPS) if name == "decoder" else ["Identity"]:
            node = ir.node(
                op,
                [value, packed] if name == "decoder" else [value],
                domain="com.microsoft" if name == "decoder" else "",
            )
            nodes.append(node)
            value = node.outputs[0]
        value.name = "y"
        value.type = ir.TensorType(ir.DataType.FLOAT)
        value.shape = ir.Shape([1, 1])
        model = ir.Model(
            ir.Graph([x], [value], nodes=nodes, initializers=[packed], opset_imports={"": 24, "com.microsoft": 1}),
            ir_version=11,
            metadata_props={"mobius.weight_loading": json.dumps(loading), "source_note": "keep this metadata"},
        )
        ir.save(model, folder / "model.onnx")
    genai = {
        "model": {
            alias: {
                "filename": f"{name}/model.onnx",
                "session_options": {"provider_options": [{"cuda": {"enable_skip_layer_norm_strict_mode": "1"}}]},
            }
            for name, alias in recipe.COMPONENTS.items()
        }
    }
    (root / "genai_config.json").write_text(json.dumps(genai))
    (root / "tokenizer.json").write_text('{"fixture": true}')
    (root / "processor_config.json").write_text('{"route": "prepared-features"}')
    return root


def test_config_schema_and_component_resolution(package):
    template = RunConfig.model_validate(json.loads(recipe.TEMPLATE.read_text()))
    assert template.passes == {}
    assert template.engine.evaluate_input_model is False
    config, inventory, files = recipe.inspect_package(package)
    handler = ModelConfig.model_validate(config.to_json()).create_model()
    assert list(handler.model_component_names) == list(recipe.COMPONENTS)
    for name, component in handler.get_model_components():
        assert Path(component.model_path) == package / name / "model.onnx"
        assert component.inference_settings["provider_options"] == [{"enable_skip_layer_norm_strict_mode": "1"}]
        assert inventory[name]["external_tensors"] == 1
        assert (
            component.model_attributes["source_revision"]
            == template.input_model.config["model_attributes"]["source_revision"]
        )
        assert component.model_attributes["graph_metadata"][name]["source_note"] == "keep this metadata"
        assert (
            component.model_attributes["genai_config"]["model"][recipe.COMPONENTS[name]]["filename"]
            == f"{name}/model.onnx"
        )
    assert {str(path.relative_to(package)) for path in files} == {
        "genai_config.json",
        "tokenizer.json",
        "processor_config.json",
        *(f"{name}/{file}" for name in recipe.COMPONENTS for file in ("model.onnx", "model.onnx.data")),
    }


def test_preserve_packed_payloads_and_metadata_without_loading_tensors(package, tmp_path, monkeypatch):
    before = {str(p.relative_to(package)): p.read_bytes() for p in package.rglob("*") if p.is_file()}
    monkeypatch.setattr(ir.ExternalTensor, "numpy", Mock(side_effect=AssertionError("must not load payloads")))
    monkeypatch.setattr(recipe, "check_runtime", Mock(side_effect=AssertionError("inspect-only skips runtime")))
    output = tmp_path / "reference"
    receipt = recipe.preserve(package, output, inspect_only=True)
    assert receipt["runtime"]["status"] == "NOT_CHECKED"
    assert receipt["optimization"] == "NONE"
    assert receipt["accuracy"] == "NOT_EVALUATED"
    assert receipt["model_readiness"] == "NOT_ESTABLISHED"
    assert set(receipt["components"]["decoder"]["operators"]) == {f"com.microsoft::{op}" for op in recipe.PACKED_OPS}
    assert {str(p.relative_to(package)): p.read_bytes() for p in package.rglob("*") if p.is_file()} == before
    serialized = ModelConfig.model_validate_json((output / "model_config.json").read_text())
    for _, component in serialized.create_model().get_model_components():
        assert component.model_attributes["precision_policy"].startswith("preserve-native")
        assert str(package / "tokenizer.json") in component.model_attributes["additional_files"]
    for relative, entry in receipt["files"].items():
        assert entry["sha256"] == recipe.sha256(package / relative)
    assert sorted(p.name for p in output.iterdir()) == ["model_config.json", "preservation_receipt.json"]


@pytest.mark.parametrize("relative", ["decoder/model.onnx", "decoder/model.onnx.data", "genai_config.json"])
def test_missing_component_or_payload_fails(package, relative):
    (package / relative).unlink()
    with pytest.raises(ValueError, match="Missing package file"):
        recipe.inspect_package(package)


def test_truncated_payload_fails(package):
    (package / "decoder/model.onnx.data").write_bytes(b"short")
    with pytest.raises(ValueError, match="outside payload bounds"):
        recipe.inspect_package(package)


def test_external_payload_symlink_escape_fails(package, tmp_path):
    outside = tmp_path / "outside.data"
    outside.write_bytes(bytes(range(128)))
    payload = package / "decoder/model.onnx.data"
    payload.unlink()
    payload.symlink_to(outside)
    with pytest.raises(ValueError, match="stay inside"):
        recipe.inspect_package(package)


def test_removed_packed_operator_fails(package):
    path = package / "decoder/model.onnx"
    model = ir.load(path)
    model.graph[0].op_type = "Identity"
    ir.save(model, path)
    with pytest.raises(ValueError, match="must retain"):
        recipe.inspect_package(package)


@pytest.mark.parametrize("relative", ["../outside", "/absolute/path"])
def test_external_path_escape_fails(package, relative):
    with pytest.raises(ValueError, match="stay inside"):
        recipe.package_file(package, relative)


def test_genai_filename_mismatch_fails(package):
    path = package / "genai_config.json"
    config = json.loads(path.read_text())
    config["model"]["decoder"]["filename"] = "embedding/model.onnx"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="filename mismatch"):
        recipe.inspect_package(package)


def test_non_cuda_genai_provider_fails(package):
    path = package / "genai_config.json"
    config = json.loads(path.read_text())
    config["model"]["decoder"]["session_options"]["provider_options"] = [{"cpu": {}}]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Only CUDA provider options"):
        recipe.inspect_package(package)


@pytest.mark.parametrize("route", ["ort-genai", "ort-extensions", "cpu"])
def test_unsupported_route_fails(package, route):
    with pytest.raises(ValueError, match="Only prepared-features"):
        recipe.inspect_package(package, route=route)


def test_requantization_pass_fails(package, tmp_path):
    config = json.loads(recipe.TEMPLATE.read_text())
    config["passes"] = {"requantize": {"type": "OnnxMatMul4Quantizer"}}
    path = tmp_path / "requantize.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="forbids passes"):
        recipe.inspect_package(package, path)


def test_native_provenance_mismatch_fails(package):
    path = package / "embedding/model.onnx"
    model = ir.load(path)
    model.metadata_props["mobius.weight_loading"] = "{}"
    ir.save(model, path)
    with pytest.raises(ValueError, match="weight-loading provenance"):
        recipe.inspect_package(package)


def test_output_in_source_fails(package):
    with pytest.raises(ValueError, match="outside the read-only"):
        recipe.preserve(package, package / "output", inspect_only=True)


def test_runtime_without_cuda_fails(package, monkeypatch):
    import onnxruntime as ort

    config, _, _ = recipe.inspect_package(package)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        recipe.check_runtime(config)


def test_missing_custom_op_fails_instead_of_requantizing(package, monkeypatch):
    import onnxruntime as ort

    config, _, _ = recipe.inspect_package(package)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(ONNXModelHandler, "prepare_session", Mock(side_effect=RuntimeError("unregistered packed op")))
    with pytest.raises(RuntimeError, match="requantization is not a fallback"):
        recipe.check_runtime(config)


def test_cuda_initialization_fallback_fails(package, monkeypatch):
    import onnxruntime as ort

    config, _, _ = recipe.inspect_package(package)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(
        ONNXModelHandler,
        "prepare_session",
        lambda *args, **kwargs: Mock(get_providers=lambda: ["CPUExecutionProvider"]),
    )
    with pytest.raises(RuntimeError, match="Native CUDA session creation failed"):
        recipe.check_runtime(config)


def test_runtime_loads_all_components_without_claiming_accuracy(package, monkeypatch):
    import onnxruntime as ort

    config, _, _ = recipe.inspect_package(package)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider"])
    prepare = Mock(return_value=Mock(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    monkeypatch.setattr(ONNXModelHandler, "prepare_session", prepare)
    result = recipe.check_runtime(config)
    assert prepare.call_count == 3
    assert result["status"] == "SESSION_CREATION_ONLY"
    assert all(
        call.kwargs == {"device": Device.GPU, "execution_providers": ["CUDAExecutionProvider"]}
        for call in prepare.call_args_list
    )
