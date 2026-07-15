# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from copy import deepcopy
from pathlib import Path
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
        assert output_dirs == [Path("out/first").resolve(), Path("out/second").resolve()]

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
        with pytest.raises(ValueError, match="select_components is only supported"):
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
        with pytest.raises(ValueError, match="Unknown component"):
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

    def test_builds_prevalidate_all_configs_before_running(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "valid": {"pipeline": ["convert"], "output_dir": "out/valid"},
            "broken": {
                "pipeline": ["convert"],
                "output_dir": "out/broken",
                "search_strategy": True,
            },
        }

        with (
            patch("olive.engine.engine.Engine.run") as run_mock,
            pytest.raises(ValueError, match="Invalid build 'broken'"),
        ):
            olive_run(config)
        run_mock.assert_not_called()

    def test_builds_union_required_packages_across_expanded_configs(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "first": {"pipeline": ["convert"], "output_dir": "out/first"},
            "second": {"pipeline": ["tune"], "output_dir": "out/second"},
        }
        run_module = sys.modules[olive_run.__module__]

        with (
            patch.object(run_module, "get_required_packages", side_effect=[{"package-a"}, {"package-b"}]),
            patch.object(run_module, "generate_files_from_packages") as generate_mock,
        ):
            olive_run(config, list_required_packages=True)

        generate_mock.assert_called_once_with({"package-a", "package-b"}, "olive_requirements.txt")

    def test_builds_use_standard_docker_workflow_dispatch(self):
        config = deepcopy(self.template)
        config["systems"]["docker_system"] = {
            "type": "Docker",
            "dockerfile": "Dockerfile",
            "build_context_path": ".",
        }
        config["builds"] = {
            "docker": {
                "pipeline": ["convert"],
                "output_dir": "out/docker",
                "host": "docker_system",
            },
        }
        docker_system = MagicMock()
        docker_output = MagicMock(name="docker_output")
        docker_system.run_workflow.return_value = docker_output

        with patch("olive.systems.system_config.SystemConfig.create_system", return_value=docker_system):
            result = olive_run(config)

        assert result == {"docker": docker_output}
        docker_run_config = docker_system.run_workflow.call_args.args[0]
        assert docker_run_config.workflow_id == "default_workflow_docker"
