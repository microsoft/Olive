# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Helpers for obtaining a model's component plan from mobius.

Mobius owns the per-architecture knowledge of which components a model exposes (e.g. a VLM's
``decoder`` / ``vision_encoder`` / ``embedding``), how each maps back to a submodule, and the role
of each component. Olive consumes that plan to drive per-component builds without re-implementing the
architecture-specific logic.

``mobius-ai`` is imported lazily so Olive keeps working when it is not installed; only the code paths
that actually need a component plan for a Hugging Face model require it.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ComponentInfo:
    """A single component returned by a component source.

    Attributes:
        name: Stable, user-facing component name used in ``builds.components``.
        kind: Component role/kind (e.g. ``decoder``, ``vision_encoder``). Optional; used for
            pass/component compatibility validation.
        source_path: Dotted submodule path locating the component inside the full model
            (e.g. ``model.language_model``). Used to slice the component for PyTorch-stage passes.

    """

    name: str
    kind: Optional[str] = None
    source_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: "ComponentInfo | dict") -> "ComponentInfo":
        if isinstance(data, ComponentInfo):
            return data
        source = data.get("source") or {}
        return cls(
            name=data["name"],
            kind=data.get("kind"),
            source_path=data.get("source_path") or source.get("path"),
            metadata={k: v for k, v in data.items() if k not in ("name", "kind", "source", "source_path")},
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
    components = [ComponentInfo.from_dict(c) for c in raw_components]
    logger.debug("mobius.inspect_components(%s) -> %s", model_name_or_path, [c.name for c in components])
    return components
