# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy

import pytest
from pydantic import ValidationError

from olive.workflows.run.builds import expand_builds, parse_run_config
from olive.workflows.run.config import RunConfig

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
                "other_system": {"type": "LocalSystem", "accelerators": [{"device": "gpu"}]},
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

    def test_builds_default_values_are_fully_replaced(self):
        expanded = self._expand(
            {
                "_default": {
                    "pipeline": ["convert", "tune"],
                    "output_dir": "out/default",
                    "host": "local_system",
                },
                "override": {
                    "pipeline": ["convert"],
                    "output_dir": "out/override",
                    "host": "other_system",
                },
                "inherit": {},
            }
        )

        assert list(expanded) == ["override", "inherit"]
        assert list(expanded["override"]["passes"]) == ["convert"]
        assert expanded["override"]["engine"]["output_dir"] == "out/override"
        assert expanded["override"]["engine"]["host"] == "other_system"
        assert list(expanded["inherit"]["passes"]) == ["convert", "tune"]
        assert expanded["inherit"]["engine"]["output_dir"] == "out/default"
        assert expanded["inherit"]["engine"]["host"] == "local_system"

    def test_builds_assign_unique_workflow_ids(self):
        expanded = self._expand(
            {
                "first": {"pipeline": ["convert"]},
                "second": {"pipeline": ["tune"]},
            }
        )

        assert expanded["first"]["workflow_id"] == "multi_first"
        assert expanded["second"]["workflow_id"] == "multi_second"
        assert expanded["first"]["engine"]["output_dir"] == "output/first"
        assert expanded["second"]["engine"]["output_dir"] == "output/second"

    def test_builds_prevalidate_duplicate_output_dirs(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "first": {"pipeline": ["convert"], "output_dir": "out/shared"},
            "second": {"pipeline": ["tune"], "output_dir": "out/shared"},
        }

        with pytest.raises(ValueError, match="overlapping writable directories"):
            parse_run_config(config)

    def test_builds_do_not_mutate_source_config(self):
        config_dict = deepcopy(self.template)
        config_dict["builds"] = {
            "only": {"pipeline": ["convert"], "output_dir": "out/only"},
        }
        original = deepcopy(config_dict)

        expand_builds(config_dict)

        assert config_dict == original

    def test_builds_missing_pipeline_after_merge_errors(self):
        with pytest.raises(ValidationError, match="pipeline"):
            self._expand(
                {
                    "_default": {"host": "local_system"},
                    "broken": {},
                }
            )

    @pytest.mark.parametrize("builds", [None, False, [], {}, {"_default": {}}])
    def test_builds_requires_at_least_one_named_build(self, builds):
        with pytest.raises(ValueError, match="builds"):
            self._expand(builds)

    @pytest.mark.parametrize("default", [None, False, []])
    def test_builds_default_must_be_a_dictionary(self, default):
        with pytest.raises(ValueError, match=r"builds\._default"):
            self._expand(
                {
                    "_default": default,
                    "only": {"pipeline": ["convert"], "output_dir": "out/only"},
                }
            )

    @pytest.mark.parametrize(
        ("build", "field_name"),
        [
            ({"pipeline": [], "output_dir": "out/only"}, "pipeline"),
            ({"pipeline": ["convert"], "output_dir": ""}, "output_dir"),
            ({"pipeline": ["convert"], "output_dir": "out/only", "components": []}, "components"),
        ],
    )
    def test_builds_rejects_empty_values(self, build, field_name):
        with pytest.raises(ValidationError, match=field_name):
            self._expand({"only": build})

    @pytest.mark.parametrize("build_name", ["", "has space", "../escape"])
    def test_builds_rejects_unsafe_names(self, build_name):
        with pytest.raises(ValueError, match="Invalid build name"):
            self._expand({build_name: {"pipeline": ["convert"], "output_dir": "out/only"}})

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

    def test_expanded_build_uses_standard_system_reference_validation(self):
        expanded = self._expand(
            {
                "broken": {
                    "pipeline": ["convert"],
                    "output_dir": "out/broken",
                    "host": "no_such_system",
                },
            }
        )

        with pytest.raises(ValidationError, match="not found"):
            RunConfig.model_validate(expanded["broken"])

    def test_empty_default_is_noop(self):
        expanded = self._expand(
            {
                "_default": {},
                "only": {"pipeline": ["convert"], "output_dir": "out/only"},
            }
        )

        assert list(expanded) == ["only"]
        assert list(expanded["only"]["passes"]) == ["convert"]
        assert "host" not in expanded["only"]["engine"]

    def test_expanded_build_uses_standard_search_validation(self):
        expanded = self._expand(
            {
                "only": {
                    "pipeline": ["convert"],
                    "output_dir": "out/only",
                    "search_strategy": True,
                },
            }
        )

        with pytest.raises(ValidationError, match="valid evaluator"):
            RunConfig.model_validate(expanded["only"])
