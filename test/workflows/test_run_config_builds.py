# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import subprocess
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

    @staticmethod
    def _composite_source(tmp_path):
        source = tmp_path / "source"
        for component in ("decoder", "embedding"):
            component_dir = source / component
            component_dir.mkdir(parents=True)
            (component_dir / "model.onnx").write_bytes(b"onnx")
        return source

    def test_run_config_schema_includes_multi_build_fields(self):
        properties = RunConfig.model_json_schema()["properties"]

        assert "builds" in properties
        assert "max_concurrent_builds" in properties
        assert "assemble_components" not in properties

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

    def test_workflow_output_contains_default_build_outputs(self, tmp_path):
        config = deepcopy(self.template)
        config["engine"] = {"output_dir": str(tmp_path / "assembled")}
        config["builds"] = {
            "decoder": {"pipeline": ["convert"]},
            "vision": {
                "pipeline": ["convert"],
                "output_dir": str(tmp_path / "external" / "vision"),
            },
        }

        parsed = parse_run_config(config)

        assert parsed["decoder"].engine.output_dir == (tmp_path / "assembled" / "decoder").resolve()
        assert parsed["vision"].engine.output_dir == (tmp_path / "external" / "vision").resolve()

    def test_builds_preserve_source_model_and_component_selections(self, tmp_path):
        source = self._composite_source(tmp_path)
        config = deepcopy(self.template)
        config["input_model"] = {
            "type": "CompositeModel",
            "config": {"model_path": str(source)},
        }
        config["output_dir"] = str(tmp_path / "assembled")
        config["builds"] = {
            "decoder-int4": {
                "components": ["decoder"],
                "pipeline": ["convert"],
            }
        }

        parsed = parse_run_config(config)

        assert parsed.component_context.input_model.type == "compositemodel"
        assert parsed.component_context.components == {"decoder-int4": ["decoder"]}
        assert parsed.component_context.output_dir == RunConfig.model_validate(config).engine.output_dir
        assert parsed["decoder-int4"].engine.output_dir == (tmp_path / "assembled" / ".builds" / "decoder-int4")

    def test_composite_component_builds_require_explicit_output(self, tmp_path):
        source = self._composite_source(tmp_path)
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(source)}}
        config["builds"] = {"decoder": {"components": ["decoder"], "pipeline": ["convert"]}}

        with pytest.raises(ValueError, match=r"explicit engine\.output_dir"):
            parse_run_config(config)

    def test_mixed_component_and_variant_builds_are_not_component_workflow(self, tmp_path):
        source = self._composite_source(tmp_path)
        config = deepcopy(self.template)
        config["input_model"] = {
            "type": "CompositeModel",
            "config": {"model_path": str(source)},
        }
        config["builds"] = {
            "decoder-int4": {
                "components": ["decoder"],
                "pipeline": ["convert"],
            },
            "full-model": {
                "pipeline": ["convert"],
            },
        }

        parsed = parse_run_config(config)

        assert parsed.component_context is None

    @pytest.mark.parametrize("directory_type", ["workflow", "workflow_ancestor", "artifact", "cache"])
    def test_component_builds_reject_write_directories_overlapping_input(self, tmp_path, directory_type):
        source = self._composite_source(tmp_path)
        config = deepcopy(self.template)
        config["input_model"] = {
            "type": "CompositeModel",
            "config": {"model_path": str(source)},
        }
        config["output_dir"] = (
            str(source / "assembled") if directory_type == "workflow" else str(tmp_path / "assembled")
        )
        if directory_type == "workflow_ancestor":
            config["output_dir"] = str(tmp_path)
        config["builds"] = {
            "decoder": {
                "components": ["decoder"],
                "pipeline": ["convert"],
            }
        }
        if directory_type == "artifact":
            config["builds"]["decoder"]["output_dir"] = str(source / "build")
        elif directory_type == "cache":
            config["cache_dir"] = str(source / "cache")

        with pytest.raises(ValueError, match="overlaps"):
            parse_run_config(config)

    def test_component_builds_reject_overlapping_selections_before_running(self, tmp_path):
        source = self._composite_source(tmp_path)
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(source)}}
        config["output_dir"] = str(tmp_path / "assembled")
        config["builds"] = {
            "decoder-int4": {"components": ["decoder"], "pipeline": ["convert"]},
            "decoder-int8": {"components": ["decoder"], "pipeline": ["convert"]},
        }

        with pytest.raises(ValueError, match="overlapping components"):
            parse_run_config(config)

    def test_component_builds_reject_artifacts_outside_reserved_namespace(self, tmp_path):
        source = self._composite_source(tmp_path)
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(source)}}
        config["output_dir"] = str(tmp_path / "assembled")
        config["builds"] = {
            "decoder": {
                "components": ["decoder"],
                "pipeline": ["convert"],
                "output_dir": str(tmp_path / "assembled" / "decoder"),
            }
        }

        with pytest.raises(ValueError, match=r"outside \.builds"):
            parse_run_config(config)

    def test_component_builds_reject_symlinked_output_before_running(self, tmp_path):
        source = self._composite_source(tmp_path)
        output = tmp_path / "assembled"
        outside = tmp_path / "unrelated"
        output.mkdir()
        outside.mkdir()
        try:
            (output / "decoder").symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Cannot create directory symlinks: {exc}")
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(source)}}
        config["output_dir"] = str(output)
        config["builds"] = {"decoder": {"components": ["decoder"], "pipeline": ["convert"]}}

        with pytest.raises(ValueError, match="symlink or junction"):
            parse_run_config(config)
        assert not list(outside.iterdir())

    def test_component_builds_reject_source_package_symlinks(self, tmp_path):
        source = self._composite_source(tmp_path)
        outside = tmp_path / "outside.txt"
        outside.write_text("not package data", encoding="utf-8")
        try:
            (source / "embedding" / "outside.txt").symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"Cannot create file symlinks: {exc}")
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(source)}}
        config["output_dir"] = str(tmp_path / "assembled")
        config["builds"] = {"decoder": {"components": ["decoder"], "pipeline": ["convert"]}}

        with pytest.raises(ValueError, match="symlink or junction"):
            parse_run_config(config)

    @pytest.mark.parametrize("linked_path", ["source", "workflow", "artifact", "default_artifact", "cache"])
    def test_component_builds_reject_linked_roots_before_running(self, tmp_path, linked_path):
        source = self._composite_source(tmp_path)
        output = tmp_path / "assembled"
        link = tmp_path / "linked"
        target = source if linked_path == "source" else tmp_path / "other"
        target.mkdir(exist_ok=True)
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Cannot create directory symlinks: {exc}")

        config = deepcopy(self.template)
        config["input_model"] = {
            "type": "CompositeModel",
            "config": {"model_path": str(link if linked_path == "source" else source)},
        }
        config["output_dir"] = str(link if linked_path == "workflow" else output)
        config["builds"] = {"decoder": {"components": ["decoder"], "pipeline": ["convert"]}}
        if linked_path == "artifact":
            config["builds"]["decoder"]["output_dir"] = str(link)
        if linked_path == "default_artifact":
            (output / ".builds").mkdir(parents=True)
            (output / ".builds" / "decoder").symlink_to(target, target_is_directory=True)
        if linked_path == "cache":
            config["cache_dir"] = str(link)

        with pytest.raises(ValueError, match="symlink or junction"):
            parse_run_config(config)
        if linked_path != "source":
            assert not list(target.iterdir())

    @pytest.mark.skipif(os.name != "nt", reason="Windows junctions only")
    def test_component_builds_reject_junction_output_root(self, tmp_path):
        source = self._composite_source(tmp_path)
        target = tmp_path / "other"
        target.mkdir()
        junction = tmp_path / "junction"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)], check=True, capture_output=True)
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(source)}}
        config["output_dir"] = str(junction)
        config["builds"] = {"decoder": {"components": ["decoder"], "pipeline": ["convert"]}}

        with pytest.raises(ValueError, match="symlink or junction"):
            parse_run_config(config)
        assert not list(target.iterdir())

    def test_component_builds_reject_existing_package_file_before_running(self, tmp_path):
        source = self._composite_source(tmp_path)
        output = tmp_path / "assembled"
        (output / "decoder").mkdir(parents=True)
        existing = output / "decoder" / "model.onnx"
        existing.write_bytes(b"preexisting")
        config = deepcopy(self.template)
        config["input_model"] = {"type": "CompositeModel", "config": {"model_path": str(source)}}
        config["output_dir"] = str(output)
        config["builds"] = {"decoder": {"components": ["decoder"], "pipeline": ["convert"]}}

        with pytest.raises(ValueError, match="already contains package file"):
            parse_run_config(config)
        assert existing.read_bytes() == b"preexisting"

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
