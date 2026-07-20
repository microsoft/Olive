# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Helpers for obtaining a model component plan from mobius.

Mobius owns the per-architecture knowledge of which components a model exposes (e.g. a VLM's
``decoder`` / ``vision_encoder`` / ``embedding``), how each maps back to a submodule, and the role
of each component. This adapter lets Olive consume that plan without re-implementing architecture-specific logic.

``mobius-ai`` is imported lazily so Olive keeps working when it is not installed; only the code paths
that actually need a component plan for a Hugging Face model require it.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _as_path_list(value: object) -> list[str]:
    """Normalize a source-path value (tuple/list/str/None) into a list of paths."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(p) for p in value if p]


@dataclass
class ComponentInfo:
    """A single component returned by a component source.

    Mirrors the shape of mobius' ``ComponentInfo`` (``mobius.inspect_components``).

    Attributes:
        name: Stable, user-facing component name.
        role: Component optimization role (e.g. ``decoder``, ``encoder``, ``embedding``).
            Optional; used for pass/component compatibility validation.
        source_paths: Dotted submodule paths locating the component inside the full model
            (e.g. ``["model.language_model"]``). A component may span multiple disjoint
            sub-modules, so this is a list.

    """

    name: str
    role: Optional[str] = None
    source_paths: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def coerce(cls, data: "ComponentInfo | dict | object") -> "ComponentInfo":
        """Normalize a component from any source into an Olive :class:`ComponentInfo`.

        Accepts an existing Olive ``ComponentInfo`` (returned as-is), a mapping following the
        component contract, or a duck-typed object exposing ``name``/``role``/``source_paths``
        attributes (e.g. a ``mobius`` ``ComponentInfo`` dataclass). For resilience against older
        mobius releases, the legacy ``kind``/``source_path`` names are accepted as fallbacks.
        """
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            source = data.get("source") or {}
            source_paths = data.get("source_paths")
            if source_paths is None:
                source_paths = data.get("source_path") or source.get("path")
            recognized = {"name", "role", "kind", "source", "source_path", "source_paths"}
            return cls(
                name=data["name"],
                role=data.get("role") or data.get("kind"),
                source_paths=_as_path_list(source_paths),
                metadata={k: v for k, v in data.items() if k not in recognized},
            )
        source_paths = getattr(data, "source_paths", None)
        if source_paths is None:
            source_paths = getattr(data, "source_path", None)
        return cls(
            name=data.name,
            role=getattr(data, "role", None) or getattr(data, "kind", None),
            source_paths=_as_path_list(source_paths),
        )


def inspect_components(
    model_name_or_path: str,
    task: Optional[str] = None,
    trust_remote_code: bool = False,
) -> list[ComponentInfo]:
    """Return the component plan for a Hugging Face model by querying mobius.

    Args:
        model_name_or_path: Hugging Face model id or local path.
        task: Optional task hint passed to mobius.
        trust_remote_code: Whether to trust remote code when mobius loads the config.

    Returns:
        A list of :class:`ComponentInfo`. An empty list means the model is single-component
        (no separable components).

    Raises:
        ImportError: If ``mobius-ai`` is not installed.

    """
    try:
        import mobius
    except ImportError as exc:
        raise ImportError(
            "mobius-ai is required to resolve model components for a Hugging Face model. "
            "Install with: pip install mobius-ai"
        ) from exc

    raw_components = mobius.inspect_components(
        model_name_or_path,
        task=task,
        trust_remote_code=trust_remote_code,
    )
    components = [ComponentInfo.coerce(c) for c in raw_components]
    logger.debug("mobius.inspect_components(%s) -> %s", model_name_or_path, [c.name for c in components])
    return components
