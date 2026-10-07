# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared metadata helpers for package configuration updates."""

from copy import deepcopy
from pathlib import Path

PACKAGE_CONFIG_UPDATES_KEY = "package_config_updates"
COMPONENT_NAME_MAPPING_KEY = "component_name_mapping"
ORT_GENAI_CONFIG_TYPE = "ort_genai"


def register_package_config_update(
    model_attributes: dict | None,
    config_type: str,
    file_name: str,
    json_paths: list[str],
) -> dict:
    """Register package configuration sections updated by a pass."""
    attributes = model_attributes if model_attributes is not None else {}
    file_path = Path(file_name)
    if file_path.name != file_name:
        raise ValueError(f"Package configuration file name must not contain directories: {file_name!r}")
    if not json_paths or any(not path.startswith("/") for path in json_paths):
        raise ValueError("Package configuration update paths must be non-empty JSON pointers.")

    updates = deepcopy(attributes.get(PACKAGE_CONFIG_UPDATES_KEY) or [])
    for update in updates:
        if update.get("type") == config_type and update.get("file_name") == file_name:
            update["json_paths"] = list(dict.fromkeys([*(update.get("json_paths") or []), *json_paths]))
            break
    else:
        updates.append({"type": config_type, "file_name": file_name, "json_paths": list(dict.fromkeys(json_paths))})

    attributes[PACKAGE_CONFIG_UPDATES_KEY] = updates
    return attributes
