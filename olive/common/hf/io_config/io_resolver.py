# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from __future__ import annotations

import functools
import logging
from importlib.resources import files
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the config directory."""
    config_files = files("olive.assets.io_configs")
    filepath = config_files.joinpath(filename)
    content = filepath.read_text(encoding="utf-8")
    return yaml.safe_load(content) or {}


@functools.lru_cache(maxsize=1)
def _load_tasks() -> dict[str, Any]:
    """Load and cache task configurations."""
    return _load_yaml("tasks.yaml")


@functools.lru_cache(maxsize=1)
def _load_diffusers() -> dict[str, Any]:
    """Load and cache diffusers configurations."""
    return _load_yaml("diffusers.yaml")


@functools.lru_cache(maxsize=1)
def _load_defaults() -> dict[str, int]:
    """Load and cache default dimension values."""
    return _load_yaml("defaults.yaml")


def get_task_template(task: str) -> dict[str, Any] | None:
    """Get task template configuration.

    Args:
        task: Task name (e.g., "text-generation", "text-classification").

    Returns:
        Task template configuration dict, or None if not found.

    """
    tasks = _load_tasks()
    return tasks.get(task)


def get_diffusers_component_config(component_name: str) -> dict[str, Any] | None:
    """Get diffusers component configuration.

    Args:
        component_name: Component name (e.g., "text_encoder", "unet").

    Returns:
        Component configuration dict, or None if not found.

    """
    diffusers = _load_diffusers()
    components = diffusers.get("components", {})
    return components.get(component_name)


def get_default_shapes() -> dict[str, int]:
    """Get default dimension values for dummy input generation.

    Returns:
        Dict mapping dimension names to default values.

    """
    defaults = _load_defaults()
    # Exclude aliases from defaults
    return {k: v for k, v in defaults.items() if k != "aliases"}


def get_aliases() -> dict[str, list[str]]:
    """Get attribute aliases for config lookup.

    Returns:
        Dict mapping dimension names to list of possible config attribute names.

    """
    defaults = _load_defaults()
    return defaults.get("aliases", {})


def _get_nested_attr(obj, attr_path: str):
    """Get nested attribute using dot notation (e.g., 'vision_config.image_size')."""
    parts = attr_path.split(".")
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_valid_config_value(value) -> bool:
    """Check if a value is a valid config value (not a mock or other proxy object)."""
    return isinstance(value, (int, float, str, bool, list, tuple, type(None)))


def resolve_alias(config, name: str, aliases: dict[str, list[str]] | None = None) -> Any | None:
    """Resolve a config attribute using aliases for fallback names.

    Supports nested attribute paths (e.g., 'vision_config.image_size').

    Args:
        config: The config object to query.
        name: The canonical name to look up (e.g., 'hidden_size', 'num_layers').
        aliases: Optional dict of aliases. If None, loads from defaults.yaml.

    Returns:
        The config value, or None if not found.

    """
    if aliases is None:
        aliases = get_aliases()

    # Check aliases first
    if name in aliases:
        for attr in aliases[name]:
            value = _get_nested_attr(config, attr)
            if _is_valid_config_value(value) and value is not None:
                return value

    # Try direct lookup
    value = _get_nested_attr(config, name)
    if _is_valid_config_value(value):
        return value
    return None


def is_task_supported(task: str) -> bool:
    """Check if a task is supported.

    Args:
        task: Task name to check.

    Returns:
        True if the task is supported.

    """
    return get_task_template(task) is not None
