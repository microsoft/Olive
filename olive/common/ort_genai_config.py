# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared metadata helpers for ORT GenAI configuration updates."""

from copy import deepcopy
from pathlib import Path

ORT_GENAI_CONFIG_UPDATES_KEY = "ort_genai_config_updates"
COMPONENT_NAME_MAPPING_KEY = "component_name_mapping"


def register_ort_genai_config_update(
    model_attributes: dict | None,
    file_name: str,
    json_paths: list[str],
) -> dict:
    """Register ORT GenAI configuration sections updated by a pass."""
    attributes = model_attributes if model_attributes is not None else {}
    file_path = Path(file_name)
    if file_path.name != file_name:
        raise ValueError(f"ORT GenAI configuration file name must not contain directories: {file_name!r}")
    if not json_paths or any(not path.startswith("/") for path in json_paths):
        raise ValueError("ORT GenAI configuration update paths must be non-empty JSON pointers.")

    updates = deepcopy(attributes.get(ORT_GENAI_CONFIG_UPDATES_KEY) or [])
    for update in updates:
        if update.get("file_name") == file_name:
            update["json_paths"] = list(dict.fromkeys([*(update.get("json_paths") or []), *json_paths]))
            break
    else:
        updates.append({"file_name": file_name, "json_paths": list(dict.fromkeys(json_paths))})

    attributes[ORT_GENAI_CONFIG_UPDATES_KEY] = updates
    return attributes
