# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

"""Utilities for reading and accumulating olive_run_metadata.json sidecars."""

import importlib.metadata
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

METADATA_FILENAME = "olive_run_metadata.json"


def _get_ran_with() -> dict[str, str]:
    """Capture runtime environment versions for the current pass invocation."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}

    try:
        from qairt import __sdk_version__ as sdk_version

        versions["qairt_sdk"] = sdk_version
    except (ImportError, AttributeError):
        # qairt not installed or version attribute not exposed
        pass

    try:
        versions["qairt_dev"] = importlib.metadata.version("qairt-dev")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        # qairt-dev not installed
        pass

    return versions


def load_metadata(model_attributes: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Load existing olive_run_metadata.json from a model's additional_files, if present."""
    if not model_attributes:
        return {}

    for filepath in model_attributes.get("additional_files", []):
        if Path(filepath).name == METADATA_FILENAME:
            try:
                with open(filepath) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to read %s: %s", filepath, e)

    return {}


def write_metadata(
    metadata: dict[str, Any],
    output_model_path: str,
    output_model_attributes: dict[str, Any],
) -> None:
    """Write metadata to olive_run_metadata.json in output_model_path and register it in additional_files."""
    output_path = Path(output_model_path)
    out_dir = output_path if output_path.is_dir() else output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / METADATA_FILENAME
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=4)

    existing = set(output_model_attributes.get("additional_files", []))
    # Remove any prior entry for this filename (e.g. carried forward from input model)
    existing = {p for p in existing if Path(p).name != METADATA_FILENAME}
    existing.add(str(out_path))
    output_model_attributes["additional_files"] = sorted(existing)


def append_pass_entry(
    metadata: dict[str, Any],
    pass_name: str,
    pass_type: str,
    recipe_dir: Optional[str] = None,
) -> None:
    """Append a pass entry with ran_with versions into metadata, initialising from recipe_metadata if first pass."""
    if "recipe_metadata" not in metadata and recipe_dir is not None:
        recipe_metadata = _load_recipe_metadata(recipe_dir)
        if recipe_metadata:
            metadata["recipe_metadata"] = recipe_metadata

    passes = metadata.setdefault("passes", [])
    entry: dict[str, Any] = {"name": pass_name, "type": pass_type, "ran_with": _get_ran_with()}
    passes.append(entry)

    # Compute validation_delta against validated_with if present; scoped to this entry
    validated_with = metadata.get("recipe_metadata", {}).get("validated_with")
    if validated_with:
        ran_with = entry["ran_with"]
        entry["validation_delta"] = {
            k: {
                "validated_with": validated_with[k],
                "ran_with": ran_with.get(k),
                "match": ran_with.get(k) == validated_with[k] if ran_with.get(k) is not None else None,
            }
            for k in validated_with
        }


def _load_recipe_metadata(recipe_dir: str) -> Optional[dict[str, Any]]:
    """Load recipe_metadata from info.yml in the recipe directory.

    Reads version and validated_with from the top level of info.yml. If info.yml
    is absent, logs a warning and returns None — ran_with is still captured per pass.
    """
    try:
        info_path = Path(recipe_dir) / "info.yml"
        if not info_path.exists():
            info_path = Path(recipe_dir) / "info.yaml"
        if not info_path.exists():
            logger.warning(
                "No info.yml found in %s; version and validated_with will not be captured in run metadata.",
                recipe_dir,
            )
            return None

        with open(info_path) as f:
            info = yaml.safe_load(f)

        metadata = {}
        if "version" in info:
            metadata["version"] = info["version"]
        if "validated_with" in info:
            metadata["validated_with"] = info["validated_with"]
        return metadata or None
    except Exception as e:
        logger.warning("Could not read recipe_metadata from info.yml in %s: %s", recipe_dir, e)
    return None
