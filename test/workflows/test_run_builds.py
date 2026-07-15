# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from olive.workflows import run as olive_run
from test.utils import get_pytorch_model_io_config, pytorch_model_loader

# pylint: disable=attribute-defined-outside-init

PT_MODEL = {
    "type": "PyTorchModel",
    "config": {
        "model_loader": pytorch_model_loader,
        "io_config": get_pytorch_model_io_config(),
    },
}


class TestRunBuilds:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.cache_dir = tmp_path / "cache"
        self.template = {
            "input_model": PT_MODEL,
            "systems": {
                "cpu_system": {"type": "LocalSystem", "accelerators": [{"device": "cpu"}]},
                "gpu_system": {"type": "LocalSystem", "accelerators": [{"device": "gpu"}]},
            },
            "passes": {
                "convert": {"type": "OnnxConversion"},
                "tune": {"type": "OrtSessionParamsTuning"},
            },
            "engine": {
                "evaluate_input_model": False,
                "cache_dir": str(self.cache_dir),
            },
        }

    def _patch_engine_and_acc(self):
        run_mock = MagicMock(return_value=MagicMock(name="WorkflowOutput"))
        acc_mock = MagicMock(name="accelerator_spec")
        engine_run_patch = patch("olive.engine.engine.Engine.run", run_mock)
        accelerator_patch = patch.object(sys.modules[olive_run.__module__], "create_accelerator", return_value=acc_mock)
        return run_mock, acc_mock, engine_run_patch, accelerator_patch

    def test_builds_passes_per_build_pipeline_subset_in_declared_order(self):
        # `tune` declared first in passes but second in pipeline; engine should receive [convert, tune].
        run_mock, _, engine_run_patch, acc_patch = self._patch_engine_and_acc()
        captured: list = []

        def capture_input_passes(self, pass_configs):
            captured.append(list(pass_configs))

        config = deepcopy(self.template)
        config["passes"] = {
            "tune": {"type": "OrtSessionParamsTuning"},
            "convert": {"type": "OnnxConversion"},
        }
        config["builds"] = {
            "only": {"pipeline": ["convert", "tune"], "output_dir": "out/only"},
        }
        with (
            engine_run_patch,
            acc_patch,
            patch("olive.engine.engine.Engine.set_input_passes_configs", capture_input_passes),
        ):
            olive_run(config)
        assert run_mock.call_count == 1
        assert captured == [["convert", "tune"]]

    def test_builds_uses_per_build_output_dir(self):
        run_mock, _, engine_run_patch, acc_patch = self._patch_engine_and_acc()
        config = deepcopy(self.template)
        config["builds"] = {
            "first": {"pipeline": ["convert"], "output_dir": "out/first"},
            "second": {"pipeline": ["convert"], "output_dir": "out/second"},
        }
        with engine_run_patch, acc_patch:
            olive_run(config)
        output_dirs = [call.args[3] for call in run_mock.call_args_list]
        assert output_dirs == ["out/first", "out/second"]

    def test_builds_host_target_override_applied_per_build(self):
        # Captures the SystemConfig passed to create_accelerator for each build.
        run_mock, acc_mock, engine_run_patch, _ = self._patch_engine_and_acc()
        seen_targets: list = []

        def fake_create_accelerator(system_config, **kwargs):
            seen_targets.append(system_config.config.accelerators[0].device.lower())
            return acc_mock

        config = deepcopy(self.template)
        config["builds"] = {
            "cpu_build": {
                "pipeline": ["convert"],
                "output_dir": "out/cpu",
                "host": "cpu_system",
                "target": "cpu_system",
            },
            "gpu_build": {
                "pipeline": ["convert"],
                "output_dir": "out/gpu",
                "host": "gpu_system",
                "target": "gpu_system",
            },
        }
        with (
            engine_run_patch,
            patch.object(sys.modules[olive_run.__module__], "create_accelerator", side_effect=fake_create_accelerator),
        ):
            olive_run(config)
        assert run_mock.call_count == 2
        assert seen_targets == ["cpu", "gpu"]

    def test_builds_components_on_non_composite_input_raises(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "broken": {
                "pipeline": ["convert"],
                "output_dir": "out/broken",
                "components": ["text_encoder"],
            },
        }
        with pytest.raises(ValueError, match="no selectable components"):
            olive_run(config)

    def test_builds_components_unknown_name_raises(self):
        composite_input = {
            "type": "CompositeModel",
            "config": {
                "model_components": [
                    {"type": "ONNXModel", "config": {"model_path": "a.onnx"}},
                    {"type": "ONNXModel", "config": {"model_path": "b.onnx"}},
                ],
                "model_component_names": ["text_encoder", "unet"],
            },
        }
        config = deepcopy(self.template)
        config["input_model"] = composite_input
        config["builds"] = {
            "bad": {
                "pipeline": ["convert"],
                "output_dir": "out/bad",
                "components": ["no_such_component"],
            },
        }
        with pytest.raises(ValueError, match="unknown component"):
            olive_run(config)

    def test_builds_directory_composite_input_runs_per_component(self, tmp_path):
        # Flow A Option 2: a mobius export directory loads as a CompositeModel,
        # subfolder names become component names, sibling builds optimize each.
        for name in ["decoder", "vision_encoder"]:
            comp_dir = tmp_path / "exported_pkg" / name
            comp_dir.mkdir(parents=True)
            (comp_dir / "model.onnx").write_bytes(b"onnx")

        run_mock, _, engine_run_patch, acc_patch = self._patch_engine_and_acc()
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(tmp_path / "exported_pkg")}}
        config["builds"] = {
            "decoder": {"components": ["decoder"], "pipeline": ["convert"], "output_dir": "out/decoder"},
            "vision_encoder": {"components": ["vision_encoder"], "pipeline": ["convert"], "output_dir": "out/vision"},
        }
        with engine_run_patch, acc_patch:
            result = olive_run(config)
        assert set(result) == {"decoder", "vision_encoder"}
        assert run_mock.call_count == 2
