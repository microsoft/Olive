# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy

import pytest
from pydantic import ValidationError

from olive.workflows.run.config import RunConfig

# pylint: disable=attribute-defined-outside-init


class TestRunConfigBuilds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.template = {
            "input_model": {
                "type": "HfModel",
                "model_path": "dummy_model",
                "task": "dummy_task",
            },
            "systems": {
                "local_system": {"type": "LocalSystem", "accelerators": [{"device": "cpu"}]},
                "other_system": {"type": "LocalSystem", "accelerators": [{"device": "gpu"}]},
            },
            "passes": {
                "convert": {"type": "OnnxConversion"},
                "tune": {"type": "OrtSessionParamsTuning"},
            },
            "evaluate_input_model": False,
        }

    def _build_config(self, builds):
        config_dict = deepcopy(self.template)
        config_dict["builds"] = builds
        return config_dict

    def test_builds_default_pipeline_full_replace(self):
        # Lists from `_default` should be fully replaced (not concatenated) by sibling values.
        config_dict = self._build_config(
            {
                "_default": {
                    "pipeline": ["convert", "tune"],
                    "components": ["text_encoder"],
                    "output_dir": "out/default",
                },
                "override": {
                    "pipeline": ["convert"],
                    "components": ["unet"],
                    "output_dir": "out/override",
                },
                "inherit": {},
            }
        )
        run_config = RunConfig.model_validate(config_dict)
        assert run_config.builds["override"].pipeline == ["convert"]
        assert run_config.builds["override"].components == ["unet"]
        assert run_config.builds["inherit"].pipeline == ["convert", "tune"]
        assert run_config.builds["inherit"].components == ["text_encoder"]
        assert run_config.builds["inherit"].output_dir == "out/default"

    def test_builds_missing_pipeline_after_merge_errors(self):
        # If neither `_default` nor the sibling supply `pipeline`/`output_dir`, validation fails.
        config_dict = self._build_config(
            {
                "_default": {"host": "local_system"},
                "broken": {"components": ["text_encoder"]},
            }
        )
        with pytest.raises(ValidationError, match="pipeline"):
            RunConfig.model_validate(config_dict)

    def test_builds_invalid_pipeline_ref_errors(self):
        # Pass names in `pipeline` must exist in the top-level `passes` dict.
        config_dict = self._build_config(
            {
                "broken": {
                    "pipeline": ["convert", "no_such_pass"],
                    "output_dir": "out/broken",
                },
            }
        )
        with pytest.raises(ValidationError, match="unknown pass"):
            RunConfig.model_validate(config_dict)

    def test_builds_invalid_host_ref_errors(self):
        # String host/target refs must exist in the top-level `systems` dict.
        config_dict = self._build_config(
            {
                "broken": {
                    "pipeline": ["convert"],
                    "output_dir": "out/broken",
                    "host": "no_such_system",
                },
            }
        )
        with pytest.raises(ValidationError, match="unknown entry"):
            RunConfig.model_validate(config_dict)

    def test_builds_empty_default_is_noop(self):
        # `_default: {}` should validate cleanly and leave siblings unchanged.
        config_dict = self._build_config(
            {
                "_default": {},
                "only": {"pipeline": ["convert"], "output_dir": "out/only"},
            }
        )
        run_config = RunConfig.model_validate(config_dict)
        assert set(run_config.builds) == {"only"}
        assert run_config.builds["only"].pipeline == ["convert"]
        assert run_config.builds["only"].host is None

    def test_builds_search_without_evaluator_errors(self):
        # Enabling search on a build with no build- or engine-level evaluator must fail validation.
        config_dict = self._build_config(
            {
                "only": {"pipeline": ["convert"], "output_dir": "out/only", "search_strategy": True},
            }
        )
        with pytest.raises(ValidationError, match="no evaluator"):
            RunConfig.model_validate(config_dict)
