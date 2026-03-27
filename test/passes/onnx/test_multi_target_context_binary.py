# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ONNXModelHandler
from olive.model.handler.multi_target import MultiTargetModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.ep_context_packager import EPContextBinaryPackager

# region: Helpers


def _make_onnx_handler(tmp_path, name="model", model_attributes=None):
    """Create a minimal ONNXModelHandler with a dummy .onnx file on disk."""
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{name}.onnx"
    model_file.write_text("dummy")
    return ONNXModelHandler(model_path=str(model_file), model_attributes=model_attributes)


def _make_multi_target(tmp_path, target_configs):
    """Build a MultiTargetModelHandler from a list of (name, attrs) tuples."""
    targets = []
    names = []
    for name, attrs in target_configs:
        handler = _make_onnx_handler(tmp_path, name=name, model_attributes=attrs)
        targets.append(handler)
        names.append(name)
    return MultiTargetModelHandler(targets, names, model_path=tmp_path, model_attributes={})


# endregion


# ===========================================================================
# MultiTargetModelHandler unit tests
# ===========================================================================


class TestMultiTargetModelHandler:
    def test_create_multi_target_handler(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1")
        h2 = _make_onnx_handler(tmp_path, "t2")

        mt = MultiTargetModelHandler([h1, h2], ["t1", "t2"], model_path=tmp_path)

        assert mt.target_names == ["t1", "t2"]
        pairs = list(mt.get_target_models())
        assert len(pairs) == 2
        assert pairs[0][0] == "t1"
        assert pairs[1][0] == "t2"

    def test_multi_target_handler_inherits_attributes(self, tmp_path):
        """Parent-level model_attributes are merged into each target model."""
        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})

        mt = MultiTargetModelHandler(
            [h1, h2],
            ["t1", "t2"],
            model_path=tmp_path,
            model_attributes={"ep": "QNNExecutionProvider", "device": "NPU"},
        )

        for _, target in mt.get_target_models():
            # Parent attributes are merged in
            assert target.model_attributes["ep"] == "QNNExecutionProvider"
            assert target.model_attributes["device"] == "NPU"

        # Target-specific attributes are preserved
        pairs = list(mt.get_target_models())
        assert pairs[0][1].model_attributes["architecture"] == "60"
        assert pairs[1][1].model_attributes["architecture"] == "73"

    def test_multi_target_handler_to_json(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})

        mt = MultiTargetModelHandler(
            [h1, h2],
            ["t1", "t2"],
            model_path=tmp_path,
            model_attributes={"ep": "QNNExecutionProvider"},
        )

        json_dict = mt.to_json()

        assert json_dict["type"].lower() == "multitargetmodel"
        assert json_dict["config"]["target_names"] == ["t1", "t2"]
        assert len(json_dict["config"]["target_models"]) == 2
        # Parent-level "ep" is in the parent config, not duplicated in targets
        assert json_dict["config"]["model_attributes"]["ep"] == "QNNExecutionProvider"

    def test_multi_target_handler_mismatched_names_raises(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1")
        with pytest.raises(AssertionError, match="Number of target models and names must match"):
            MultiTargetModelHandler([h1], ["t1", "t2"], model_path=tmp_path)

    def test_multi_target_handler_load_model_raises(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1")
        mt = MultiTargetModelHandler([h1], ["t1"], model_path=tmp_path)
        with pytest.raises(NotImplementedError):
            mt.load_model()

    def test_multi_target_handler_prepare_session_raises(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1")
        mt = MultiTargetModelHandler([h1], ["t1"], model_path=tmp_path)
        with pytest.raises(RuntimeError, match="doesn't have a session"):
            mt.prepare_session()

    def test_multi_target_handler_run_session_raises(self, tmp_path):
        h1 = _make_onnx_handler(tmp_path, "t1")
        mt = MultiTargetModelHandler([h1], ["t1"], model_path=tmp_path)
        with pytest.raises(RuntimeError, match="doesn't have a session"):
            mt.run_session()


# ===========================================================================
# EPContextBinaryPackager tests
# ===========================================================================


class TestEPContextBinaryPackager:
    def _create_packager(self, ep="QNNExecutionProvider", device="NPU", config=None):
        accelerator_spec = AcceleratorSpec(accelerator_type=device, execution_provider=ep)
        return create_pass_from_dict(
            EPContextBinaryPackager,
            config or {},
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

    def test_packager_generates_manifest(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [
                ("soc_60", {"architecture": "60", "precision": "int4"}),
                ("soc_73", {"architecture": "73", "precision": "int4"}),
            ],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        result = p.run(mt, output_path)

        # Result is still a MultiTargetModelHandler
        assert isinstance(result, MultiTargetModelHandler)

        # manifest.json exists
        manifest_path = tmp_path / "output" / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert len(manifest["components"]) == 2
        assert manifest["components"][0]["variant_name"] == "soc_60"
        assert manifest["components"][0]["constraints"]["architecture"] == "60"
        assert manifest["components"][0]["constraints"]["precision"] == "int4"
        assert manifest["components"][1]["variant_name"] == "soc_73"

    def test_packager_with_sdk_version(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [
                ("soc_60", {"architecture": "60", "sdk_version": "qnn_2.28"}),
                ("soc_73", {"architecture": "73", "sdk_version": "qnn_2.28"}),
            ],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        manifest_path = tmp_path / "output" / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["components"][0]["constraints"]["sdk_version"] == "qnn_2.28"

    def test_packager_sdk_version_from_config(self, tmp_path):
        """sdk_version from pass config is used when model_attributes doesn't have it."""
        mt = _make_multi_target(
            tmp_path,
            [("soc_60", {"architecture": "60"}), ("soc_73", {"architecture": "73"})],
        )

        p = self._create_packager(config={"sdk_version": "qnn_2.30"})
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        manifest_path = tmp_path / "output" / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["components"][0]["constraints"]["sdk_version"] == "qnn_2.30"

    def test_packager_compile_options(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("soc_60", {"architecture": "60"}), ("soc_73", {"architecture": "73"})],
        )

        p = self._create_packager(config={"compile_options": {"dynamic_shape": True}})
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        manifest_path = tmp_path / "output" / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["components"][0]["constraints"]["compile_options"] == {"dynamic_shape": True}

    def test_packager_custom_model_name(self, tmp_path):
        mt = _make_multi_target(
            tmp_path,
            [("soc_60", {}), ("soc_73", {})],
        )

        p = self._create_packager(config={"model_name": "my_model"})
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        manifest_path = tmp_path / "output" / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["name"] == "my_model"

    def test_packager_rejects_non_multi_target(self, tmp_path):
        handler = _make_onnx_handler(tmp_path, "single")
        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        with pytest.raises(AssertionError, match="requires a MultiTargetModelHandler"):
            p.run(handler, output_path)

    def test_packager_copies_files(self, tmp_path):
        """Verify that model files are actually copied to the output directory."""
        mt = _make_multi_target(
            tmp_path,
            [("soc_60", {"architecture": "60"}), ("soc_73", {"architecture": "73"})],
        )

        p = self._create_packager()
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # Check files were copied
        assert (tmp_path / "output" / "soc_60").is_dir()
        assert (tmp_path / "output" / "soc_73").is_dir()


# ===========================================================================
# Pass.run() multi-target auto-dispatch tests
# ===========================================================================


class TestPassRunMultiTarget:
    def test_pass_run_iterates_targets(self, tmp_path):
        """A pass that does NOT accept multi-target should iterate over each target independently."""
        from olive.passes.onnx.float16_conversion import OnnxFloatToFloat16

        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})
        mt = MultiTargetModelHandler([h1, h2], ["t1", "t2"], model_path=tmp_path)

        accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")

        # Mock _run_for_config to just return a new handler (avoid real ONNX ops)
        with patch.object(OnnxFloatToFloat16, "_run_for_config") as mock_run:

            def side_effect(model, config, output_model_path):
                out_file = Path(output_model_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text("dummy")
                return ONNXModelHandler(model_path=str(out_file), model_attributes=model.model_attributes)

            mock_run.side_effect = side_effect

            p = create_pass_from_dict(OnnxFloatToFloat16, {}, disable_search=True, accelerator_spec=accelerator_spec)
            output_path = str(tmp_path / "output.onnx")
            result = p.run(mt, output_path)

        # Result should still be MultiTargetModelHandler
        assert isinstance(result, MultiTargetModelHandler)
        assert result.target_names == ["t1", "t2"]
        # _run_for_config was called twice, once per target
        assert mock_run.call_count == 2


# ===========================================================================
# QNN EPContextBinaryGenerator multi-target tests (mocked)
# ===========================================================================


class TestQNNMultiTarget:
    @staticmethod
    def _mock_get_available_providers():
        return ["QNNExecutionProvider", "CPUExecutionProvider"]

    def test_run_multi_target_returns_multi_target_handler(self, tmp_path):
        """When provider_options is a list, result should be MultiTargetModelHandler."""
        from olive.passes.onnx.context_binary import EPContextBinaryGenerator

        accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")

        p = create_pass_from_dict(
            EPContextBinaryGenerator,
            {
                "provider_options": [
                    {"soc_model": "60", "htp_performance_mode": "burst"},
                    {"soc_model": "73", "htp_performance_mode": "burst"},
                ],
            },
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        # Mock _run_single_target to avoid real QNN invocation, and get_available_providers
        with (
            patch.object(EPContextBinaryGenerator, "_run_single_target") as mock_single,
            patch("onnxruntime.get_available_providers", self._mock_get_available_providers),
        ):

            def side_effect(model, config, output_model_path):
                out_dir = Path(output_model_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                model_file = out_dir / "model_ctx.onnx"
                model_file.write_text("dummy")
                return ONNXModelHandler(model_path=str(model_file))

            mock_single.side_effect = side_effect

            input_model = _make_onnx_handler(tmp_path, "input")
            output_path = str(tmp_path / "output.onnx")
            result = p.run(input_model, output_path)

        assert isinstance(result, MultiTargetModelHandler)
        assert result.target_names == ["soc_60", "soc_73"]
        assert mock_single.call_count == 2

        # Check model_attributes on targets
        for _, target in result.get_target_models():
            assert target.model_attributes["ep"] == "QNNExecutionProvider"
            assert target.model_attributes["device"] == "NPU"
            assert "provider_options" in target.model_attributes

    def test_run_single_target_populates_model_attributes(self, tmp_path):
        """Single-target mode should also populate model_attributes."""
        from olive.passes.onnx.context_binary import EPContextBinaryGenerator

        accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")

        p = create_pass_from_dict(
            EPContextBinaryGenerator,
            {
                "provider_options": {
                    "soc_model": "60",
                    "htp_performance_mode": "burst",
                },
            },
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        with (
            patch.object(EPContextBinaryGenerator, "_run_single_target") as mock_single,
            patch("onnxruntime.get_available_providers", self._mock_get_available_providers),
        ):

            def side_effect(model, config, output_model_path):
                out_path = Path(output_model_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("dummy")
                return ONNXModelHandler(model_path=str(out_path))

            mock_single.side_effect = side_effect

            input_model = _make_onnx_handler(tmp_path, "input")
            output_path = str(tmp_path / "output.onnx")
            result = p.run(input_model, output_path)

        assert isinstance(result, ONNXModelHandler)
        assert result.model_attributes["ep"] == "QNNExecutionProvider"
        assert result.model_attributes["device"] == "NPU"
        assert result.model_attributes["architecture"] == "60"
        assert result.model_attributes["provider_options"]["soc_model"] == "60"


# ===========================================================================
# OpenVINO multi-target tests (mocked)
# ===========================================================================


class TestOpenVINOMultiTarget:
    def test_run_multi_target_returns_multi_target_handler(self, tmp_path):
        """When ov_version is a list, result should be MultiTargetModelHandler."""
        from olive.passes.openvino.encapsulation import OpenVINOEncapsulation

        accelerator_spec = AcceleratorSpec(accelerator_type=Device.NPU, execution_provider="OpenVINOExecutionProvider")

        p = create_pass_from_dict(
            OpenVINOEncapsulation,
            {"ov_version": ["2025.1", "2025.2"], "target_device": "npu"},
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        with patch.object(OpenVINOEncapsulation, "_run_single_target") as mock_single:

            def side_effect(model, config, output_model_path):
                out_dir = Path(output_model_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                model_file = out_dir / "model.onnx"
                model_file.write_text("dummy")
                return ONNXModelHandler(
                    model_path=str(model_file),
                    model_attributes={
                        "ep": "OpenVINOExecutionProvider",
                        "device": "NPU",
                        "sdk_version": config.ov_version,
                        "architecture": "NPU",
                    },
                )

            mock_single.side_effect = side_effect

            input_model = MagicMock()
            input_model.model_attributes = {}
            output_path = str(tmp_path / "output.onnx")
            result = p.run(input_model, output_path)

        assert isinstance(result, MultiTargetModelHandler)
        assert result.target_names == ["ov_2025.1", "ov_2025.2"]
        assert mock_single.call_count == 2

    def test_run_single_target_populates_model_attributes(self, tmp_path):
        """Single-target mode should populate model_attributes with OV metadata."""
        from olive.passes.openvino.encapsulation import OpenVINOEncapsulation

        accelerator_spec = AcceleratorSpec(accelerator_type=Device.NPU, execution_provider="OpenVINOExecutionProvider")

        p = create_pass_from_dict(
            OpenVINOEncapsulation,
            {"ov_version": "2025.1", "target_device": "npu"},
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

        with patch.object(OpenVINOEncapsulation, "_run_single_target") as mock_single:

            def side_effect(model, config, output_model_path):
                out_dir = Path(output_model_path)
                out_dir.parent.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                model_file = out_dir / "model.onnx"
                model_file.write_text("dummy")
                return ONNXModelHandler(
                    model_path=str(model_file),
                    model_attributes={
                        "ep": "OpenVINOExecutionProvider",
                        "device": "NPU",
                        "sdk_version": "2025.1",
                        "architecture": "NPU",
                    },
                )

            mock_single.side_effect = side_effect

            input_model = MagicMock()
            input_model.model_attributes = {}
            output_path = str(tmp_path / "output.onnx")
            result = p.run(input_model, output_path)

        assert isinstance(result, ONNXModelHandler)
        assert result.model_attributes["ep"] == "OpenVINOExecutionProvider"
        assert result.model_attributes["sdk_version"] == "2025.1"
