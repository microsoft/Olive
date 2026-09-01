# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from olive.workflows.run.builds import MultiBuildRunConfig, expand_builds, parse_run_config
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

    def test_run_config_schema_includes_multi_build_fields(self):
        properties = RunConfig.model_json_schema()["properties"]

        assert "builds" in properties
        assert "max_concurrent_builds" in properties
        assert "assemble_components" in properties

    def test_ordinary_run_config_with_null_builds_round_trips(self):
        config = deepcopy(self.template)
        config["builds"] = None
        config["max_concurrent_builds"] = None

        parsed = parse_run_config(config)
        serialized = parsed.to_json()

        assert isinstance(parsed, RunConfig)
        assert not isinstance(parsed, MultiBuildRunConfig)
        assert "builds" not in serialized
        assert "max_concurrent_builds" not in serialized
        assert "assemble_components" not in serialized
        assert isinstance(parse_run_config(serialized), RunConfig)

    def test_builds_prevalidate_duplicate_output_dirs(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "first": {"pipeline": ["convert"], "output_dir": "out/shared"},
            "second": {"pipeline": ["tune"], "output_dir": "out/shared"},
        }

        with pytest.raises(ValueError, match="overlapping writable directories"):
            parse_run_config(config)

    def test_builds_treat_dotted_default_output_as_directory(self):
        config = deepcopy(self.template)
        config["builds"] = {
            "llama.q4": {"pipeline": ["convert"]},
            "plain": {"pipeline": ["tune"]},
        }

        parsed = parse_run_config(config)

        assert parsed["llama.q4"].engine.output_dir == (Path.cwd() / "output" / "llama.q4").resolve()
        assert parsed["plain"].engine.output_dir == (Path.cwd() / "output" / "plain").resolve()

    def test_builds_default_output_dir_is_parent(self, tmp_path):
        config = deepcopy(self.template)
        config["builds"] = {
            "_default": {"pipeline": ["convert"], "output_dir": str(tmp_path / "shared-root")},
            "first": {},
            "second": {},
            "custom": {"output_dir": str(tmp_path / "custom")},
        }

        parsed = parse_run_config(config)

        assert parsed["first"].engine.output_dir == (tmp_path / "shared-root" / "first").resolve()
        assert parsed["second"].engine.output_dir == (tmp_path / "shared-root" / "second").resolve()
        assert parsed["custom"].engine.output_dir == (tmp_path / "custom").resolve()
        assert parsed.assembly_output_dir is None

    def test_builds_record_shared_default_output_for_explicit_assembly(self, tmp_path):
        config = deepcopy(self.template)
        config["assemble_components"] = True
        config["builds"] = {
            "_default": {"pipeline": ["convert"], "output_dir": str(tmp_path / "assembled")},
            "decoder": {},
            "vision": {},
        }

        parsed = parse_run_config(config)

        assert parsed.assembly_output_dir == (tmp_path / "assembled").resolve()

    def test_builds_do_not_assemble_by_default(self, tmp_path):
        config = deepcopy(self.template)
        config["builds"] = {
            "_default": {"pipeline": ["convert"], "output_dir": str(tmp_path / "assembled")},
            "decoder": {},
            "vision": {},
        }

        parsed = parse_run_config(config)

        assert parsed.assembly_output_dir is None
        assert parsed.assemble_components is False

    def test_builds_derive_assembly_parent_from_explicit_sibling_outputs(self, tmp_path):
        config = deepcopy(self.template)
        config["assemble_components"] = True
        config["builds"] = {
            "decoder": {
                "pipeline": ["convert"],
                "output_dir": str(tmp_path / "assembled" / "decoder-int4"),
            },
            "vision": {
                "pipeline": ["convert"],
                "output_dir": str(tmp_path / "assembled" / "vision-int4"),
            },
        }

        parsed = parse_run_config(config)

        assert parsed.assembly_output_dir == (tmp_path / "assembled").resolve()

    def test_builds_can_disable_automatic_assembly(self, tmp_path):
        config = deepcopy(self.template)
        config["assemble_components"] = False
        config["builds"] = {
            "_default": {"pipeline": ["convert"], "output_dir": str(tmp_path / "assembled")},
            "decoder": {},
            "vision": {},
        }

        parsed = parse_run_config(config)

        assert parsed.assembly_output_dir is None
        assert parsed.assemble_components is False

    def test_builds_required_assembly_rejects_different_parents(self, tmp_path):
        config = deepcopy(self.template)
        config["assemble_components"] = True
        config["builds"] = {
            "decoder": {
                "pipeline": ["convert"],
                "output_dir": str(tmp_path / "decoder" / "model"),
            },
            "vision": {
                "pipeline": ["convert"],
                "output_dir": str(tmp_path / "vision" / "model"),
            },
        }

        with pytest.raises(ValueError, match="same parent directory"):
            parse_run_config(config)

    @pytest.mark.parametrize("assemble_components", [None, 0, 1, "true", [], {}])
    def test_builds_reject_invalid_assemble_components(self, assemble_components):
        config = deepcopy(self.template)
        config["assemble_components"] = assemble_components
        config["builds"] = {
            "only": {"pipeline": ["convert"], "output_dir": "out/only"},
        }

        with pytest.raises(ValueError, match="must be true or false"):
            parse_run_config(config)

    @pytest.mark.parametrize("max_concurrent_builds", [None, 2])
    def test_builds_parse_max_concurrent_builds(self, max_concurrent_builds):
        config = deepcopy(self.template)
        if max_concurrent_builds is not None:
            config["max_concurrent_builds"] = max_concurrent_builds
        config["builds"] = {
            "first": {"pipeline": ["convert"], "output_dir": "out/first"},
            "second": {"pipeline": ["convert"], "output_dir": "out/second"},
        }

        parsed = parse_run_config(config)

        assert isinstance(parsed, MultiBuildRunConfig)
        assert parsed.max_concurrent_builds == (max_concurrent_builds or 2)
        assert set(parsed) == {"first", "second"}

    @pytest.mark.parametrize("max_concurrent_builds", [0, -1, True, "2", 1.5])
    def test_builds_reject_invalid_max_concurrent_builds(self, max_concurrent_builds):
        config = deepcopy(self.template)
        config["max_concurrent_builds"] = max_concurrent_builds
        config["builds"] = {
            "only": {"pipeline": ["convert"], "output_dir": "out/only"},
        }

        with pytest.raises(ValueError, match="must be a positive integer"):
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
