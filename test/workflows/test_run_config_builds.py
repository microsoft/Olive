# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy

import pytest
from pydantic import ValidationError

from olive.workflows.run.builds import expand_builds, parse_run_config

# pylint: disable=attribute-defined-outside-init


class TestBuildConfigExpansion:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.template = {
            "workflow_id": "multi",
            "input_model": {
                "type": "HfModel",
                "model_path": "dummy_model",
                "task": "dummy_task",
            },
            "systems": {
                "local_system": {"type": "LocalSystem", "accelerators": [{"device": "cpu"}]},
            },
            "passes": {
                "convert": {"type": "OnnxConversion"},
                "tune": {"type": "OrtSessionParamsTuning"},
            },
            "evaluate_input_model": False,
        }

    def _expand(self, builds):
        config_dict = deepcopy(self.template)
        config_dict["builds"] = builds
        return expand_builds(config_dict)

    def test_builds_prevalidate_duplicate_output_dirs(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "first": {"pipeline": ["convert"], "output_dir": "out/shared"},
            "second": {"pipeline": ["tune"], "output_dir": "out/shared"},
        }

        with pytest.raises(ValueError, match="overlapping writable directories"):
            parse_run_config(config)

    def test_builds_missing_pipeline_after_merge_errors(self):
        with pytest.raises(ValidationError, match="pipeline"):
            self._expand(
                {
                    "_default": {"host": "local_system"},
                    "broken": {},
                }
            )

    @pytest.mark.parametrize("builds", [None, {"_default": {}}])
    def test_builds_requires_at_least_one_named_build(self, builds):
        with pytest.raises(ValueError, match="builds"):
            self._expand(builds)

    def test_builds_default_must_be_a_dictionary(self):
        with pytest.raises(ValueError, match=r"builds\._default"):
            self._expand(
                {
                    "_default": None,
                    "only": {"pipeline": ["convert"], "output_dir": "out/only"},
                }
            )

    def test_builds_rejects_unsafe_names(self):
        with pytest.raises(ValueError, match="Invalid build name"):
            self._expand({"has space": {"pipeline": ["convert"], "output_dir": "out/only"}})

    def test_builds_invalid_pipeline_reference_errors(self):
        with pytest.raises(ValueError, match="unknown pass"):
            self._expand(
                {
                    "broken": {
                        "pipeline": ["convert", "no_such_pass"],
                        "output_dir": "out/broken",
                    },
                }
            )
