# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
import time
from copy import deepcopy
from pathlib import Path
from threading import Barrier
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
        assert set(output_dirs) == {Path("out/first").resolve(), Path("out/second").resolve()}

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
        assert set(seen_targets) == {"cpu", "gpu"}

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
        # An export directory loads as a CompositeModel; subfolder names become
        # component names, and sibling builds optimize each component.
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

    @pytest.mark.parametrize("host_location", ["engine", "build"])
    def test_builds_reject_non_local_host(self, host_location):
        config = deepcopy(self.template)
        config["systems"]["docker_system"] = {
            "type": "Docker",
            "dockerfile": "Dockerfile",
            "build_context_path": ".",
        }
        config["builds"] = {
            "docker": {"pipeline": ["convert"], "output_dir": "out/docker"},
        }
        if host_location == "engine":
            config["engine"]["host"] = "docker_system"
        else:
            config["builds"]["docker"]["host"] = "docker_system"

        with (
            patch("olive.engine.engine.Engine.run") as run_mock,
            pytest.raises(ValueError, match="only LocalSystem hosts"),
        ):
            olive_run(config)
        run_mock.assert_not_called()

    def test_builds_run_in_parallel_and_preserve_configured_result_order(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "slow": {"pipeline": ["convert"], "output_dir": "out/slow"},
            "fast": {"pipeline": ["tune"], "output_dir": "out/fast"},
        }
        barrier = Barrier(2)

        def run_build(_, run_config):
            barrier.wait(timeout=2)
            if run_config.workflow_id.endswith("_slow"):
                time.sleep(0.05)
            return run_config.workflow_id

        with patch.object(sys.modules[olive_run.__module__], "_run_single", side_effect=run_build):
            result = olive_run(config)

        assert result == {
            "slow": "default_workflow_slow",
            "fast": "default_workflow_fast",
        }

    def test_builds_report_build_name_when_parallel_execution_fails(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "good": {"pipeline": ["convert"], "output_dir": "out/good"},
            "bad": {"pipeline": ["tune"], "output_dir": "out/bad"},
        }
        barrier = Barrier(2)

        def run_build(_, run_config):
            barrier.wait(timeout=2)
            if run_config.workflow_id.endswith("_bad"):
                raise ValueError("bad build")
            return run_config.workflow_id

        with (
            patch.object(sys.modules[olive_run.__module__], "_run_single", side_effect=run_build),
            pytest.raises(RuntimeError, match=r"Build\(s\) \['bad'\] failed.*bad build"),
        ):
            olive_run(config)

    def test_builds_report_all_parallel_execution_failures(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "first": {"pipeline": ["convert"], "output_dir": "out/first"},
            "second": {"pipeline": ["tune"], "output_dir": "out/second"},
        }

        def run_build(_, run_config):
            raise ValueError(f"{run_config.workflow_id} error")

        with (
            patch.object(sys.modules[olive_run.__module__], "_run_single", side_effect=run_build),
            pytest.raises(
                RuntimeError,
                match=(
                    r"Build\(s\) \['first', 'second'\] failed:.*default_workflow_first error"
                    r".*default_workflow_second error"
                ),
            ),
        ):
            olive_run(config)

    def test_builds_reject_overlapping_parallel_artifact_directories(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "parent": {"pipeline": ["convert"], "output_dir": "out/shared"},
            "child": {"pipeline": ["tune"], "output_dir": "out/shared/child"},
        }

        with (
            patch.object(sys.modules[olive_run.__module__], "_run_single") as run_mock,
            pytest.raises(ValueError, match="overlapping writable directories"),
        ):
            olive_run(config)
        run_mock.assert_not_called()

    def test_builds_reject_artifact_directory_overlapping_another_build_cache(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "artifact": {
                "pipeline": ["convert"],
                "output_dir": str(self.cache_dir / "default_workflow_cached"),
            },
            "cached": {"pipeline": ["tune"], "output_dir": "out/cached"},
        }

        with (
            patch.object(sys.modules[olive_run.__module__], "_run_single") as run_mock,
            pytest.raises(ValueError, match=r"artifact directory .* and cache directory"),
        ):
            olive_run(config)
        run_mock.assert_not_called()
