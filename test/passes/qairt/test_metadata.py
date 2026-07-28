# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import json
from pathlib import Path
from unittest.mock import patch

from olive.passes.qairt.utils.metadata import (
    METADATA_FILENAME,
    _get_ran_with,
    _load_recipe_metadata,
    append_pass_entry,
    load_metadata,
    write_metadata,
)

# ---------------------------------------------------------------------------
# _get_ran_with
# ---------------------------------------------------------------------------


def test_get_ran_with_always_has_python():
    result = _get_ran_with()
    assert "python" in result
    assert result["python"]  # non-empty


def test_get_ran_with_captures_qairt_sdk(mock_qairt_modules):
    mock_qairt_modules["qairt"].__sdk_version__ = "2.45.0"
    result = _get_ran_with()
    assert result.get("qairt_sdk") == "2.45.0"


def test_get_ran_with_missing_qairt_omits_key():
    with patch.dict("sys.modules", {"qairt": None}):
        result = _get_ran_with()
    assert "qairt_sdk" not in result


# ---------------------------------------------------------------------------
# _load_recipe_metadata
# ---------------------------------------------------------------------------


def test_load_recipe_metadata_reads_top_level_fields(tmp_path):
    (tmp_path / "info.yml").write_text(
        "version: '1.2.3'\n"
        "validated_with:\n"
        "  qairt_sdk: '2.45.40'\n"
        "  qairt_dev: '0.8.1'\n"
        "  python: '3.10.12'\n"
        "recipes:\n"
        "  - file: htp_sc8380xp.json\n"
    )
    meta = _load_recipe_metadata(str(tmp_path))
    assert meta == {
        "version": "1.2.3",
        "validated_with": {"qairt_sdk": "2.45.40", "qairt_dev": "0.8.1", "python": "3.10.12"},
    }


def test_load_recipe_metadata_falls_back_to_info_yaml(tmp_path):
    (tmp_path / "info.yaml").write_text("version: '0.9.0'\n")
    meta = _load_recipe_metadata(str(tmp_path))
    assert meta == {"version": "0.9.0"}


def test_load_recipe_metadata_warns_when_no_info_yml(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        meta = _load_recipe_metadata(str(tmp_path))
    assert meta is None
    assert "No info.yml found" in caplog.text


def test_load_recipe_metadata_returns_none_when_no_version_fields(tmp_path):
    (tmp_path / "info.yml").write_text("keywords:\n  - qairt\nrecipes: []\n")
    meta = _load_recipe_metadata(str(tmp_path))
    assert meta is None


def test_load_recipe_metadata_warns_on_parse_error(tmp_path, caplog):
    import logging

    (tmp_path / "info.yml").write_text(":\ninvalid: [yaml\n")
    with caplog.at_level(logging.WARNING):
        meta = _load_recipe_metadata(str(tmp_path))
    assert meta is None
    assert "Could not read recipe_metadata" in caplog.text


# ---------------------------------------------------------------------------
# load_metadata / write_metadata
# ---------------------------------------------------------------------------


def test_load_metadata_returns_empty_when_no_attributes():
    assert load_metadata(None) == {}
    assert load_metadata({}) == {}


def test_load_metadata_returns_empty_when_no_additional_files():
    assert load_metadata({"additional_files": []}) == {}


def test_write_and_load_metadata_roundtrip(tmp_path):
    metadata = {"recipe_metadata": {"version": "1.0.0"}, "passes": []}
    attrs: dict = {}
    write_metadata(metadata, str(tmp_path), attrs)

    assert METADATA_FILENAME in attrs["additional_files"][0]
    loaded = load_metadata(attrs)
    assert loaded == metadata


def test_write_metadata_replaces_prior_entry(tmp_path):
    old_path = tmp_path / "old" / METADATA_FILENAME
    old_path.parent.mkdir()
    old_path.write_text("{}")

    attrs = {"additional_files": [str(old_path)]}
    write_metadata({"passes": []}, str(tmp_path), attrs)

    # Only one entry for METADATA_FILENAME
    hits = [p for p in attrs["additional_files"] if Path(p).name == METADATA_FILENAME]
    assert len(hits) == 1
    assert str(tmp_path) in hits[0]


# ---------------------------------------------------------------------------
# append_pass_entry
# ---------------------------------------------------------------------------


def test_append_pass_entry_populates_ran_with():
    metadata: dict = {}
    append_pass_entry(metadata, "QairtPreparation", "QairtPreparation")
    assert len(metadata["passes"]) == 1
    entry = metadata["passes"][0]
    assert entry["name"] == "QairtPreparation"
    assert "python" in entry["ran_with"]


def test_append_pass_entry_seeds_recipe_metadata_on_first_pass(tmp_path):
    (tmp_path / "info.yml").write_text("version: '1.0.0'\nvalidated_with:\n  python: '3.10.12'\n")
    metadata: dict = {}
    append_pass_entry(metadata, "QairtPreparation", "QairtPreparation", recipe_dir=str(tmp_path))
    assert metadata["recipe_metadata"]["version"] == "1.0.0"


def test_append_pass_entry_does_not_re_seed_on_subsequent_passes(tmp_path):
    (tmp_path / "info.yml").write_text("version: '1.0.0'\n")
    metadata: dict = {"recipe_metadata": {"version": "already-set"}}
    append_pass_entry(metadata, "QairtGenAIBuilder", "QairtGenAIBuilder", recipe_dir=str(tmp_path))
    assert metadata["recipe_metadata"]["version"] == "already-set"


def test_append_pass_entry_computes_validation_delta(tmp_path):
    (tmp_path / "info.yml").write_text(
        "version: '1.0.0'\nvalidated_with:\n  python: '3.10.12'\n  qairt_sdk: '2.48.0'\n"
    )
    metadata: dict = {}
    with patch(
        "olive.passes.qairt.utils.metadata._get_ran_with", return_value={"python": "3.10.12", "qairt_sdk": "2.45.40"}
    ):
        append_pass_entry(metadata, "QairtPreparation", "QairtPreparation", recipe_dir=str(tmp_path))

    delta = metadata["passes"][0]["validation_delta"]
    assert delta["python"]["match"] is True
    assert delta["qairt_sdk"]["match"] is False


def test_append_pass_entry_no_delta_when_no_validated_with(tmp_path):
    (tmp_path / "info.yml").write_text("version: '1.0.0'\n")
    metadata: dict = {}
    append_pass_entry(metadata, "QairtPreparation", "QairtPreparation", recipe_dir=str(tmp_path))
    assert "validation_delta" not in metadata["passes"][0]


def test_append_pass_entry_no_recipe_dir_skips_metadata_seed():
    metadata: dict = {}
    append_pass_entry(metadata, "QairtPreparation", "QairtPreparation", recipe_dir=None)
    assert "recipe_metadata" not in metadata
    assert len(metadata["passes"]) == 1


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_run_config_without_vendor_field_parses_cleanly():
    """RunConfig JSON with no vendor block should parse without error on new Olive."""
    from olive.workflows.run.config import RunConfig

    config = {
        "input_model": {"type": "HfModel", "model_path": "microsoft/phi-4-mini-instruct"},
        "passes": {},
    }
    # Should not raise — vendor field is gone, no extra_forbidden issue
    run_config = RunConfig.model_validate(config)
    assert run_config.input_model is not None


def test_load_metadata_with_legacy_format_no_recipe_metadata(tmp_path):
    """A metadata file written by an older version (passes only, no recipe_metadata) loads cleanly."""
    legacy = {"passes": [{"name": "QairtPreparation", "type": "QairtPreparation", "ran_with": {"python": "3.10.12"}}]}
    meta_file = tmp_path / METADATA_FILENAME
    meta_file.write_text(json.dumps(legacy))

    attrs = {"additional_files": [str(meta_file)]}
    loaded = load_metadata(attrs)
    assert loaded == legacy
    assert "recipe_metadata" not in loaded


def test_append_pass_entry_carries_forward_legacy_metadata(tmp_path):
    """A second pass run with legacy metadata (no recipe_metadata) still appends ran_with correctly."""
    legacy = {"passes": [{"name": "QairtPreparation", "type": "QairtPreparation", "ran_with": {"python": "3.10.12"}}]}
    # No recipe_metadata in dict, no info.yml in dir — second pass should append cleanly
    append_pass_entry(legacy, "QairtGenAIBuilder", "QairtGenAIBuilder", recipe_dir=str(tmp_path))
    assert len(legacy["passes"]) == 2
    assert legacy["passes"][1]["name"] == "QairtGenAIBuilder"
    assert "recipe_metadata" not in legacy


def test_load_metadata_with_no_additional_files_key():
    """model_attributes that has no additional_files key at all returns empty dict."""
    assert load_metadata({"model_path": "/some/path"}) == {}


def test_write_metadata_with_no_prior_model_attributes(tmp_path):
    """write_metadata works correctly when model_attributes starts as an empty dict."""
    attrs: dict = {}
    write_metadata({"passes": []}, str(tmp_path), attrs)
    assert len(attrs["additional_files"]) == 1
    assert Path(attrs["additional_files"][0]).name == METADATA_FILENAME
