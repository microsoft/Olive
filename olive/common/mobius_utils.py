# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Helpers for obtaining a model component plan from mobius.

Mobius owns the per-architecture knowledge of which components a model exposes (e.g. a VLM's
``decoder`` / ``vision_encoder`` / ``embedding``), how each maps back to a submodule, and the role
of each component. This adapter lets Olive consume that plan without re-implementing architecture-specific logic.

``mobius-onnx`` is imported lazily so Olive keeps working when it is not installed; only the code paths
that actually need a component plan for a Hugging Face model require it.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _as_path_list(value: object) -> list[str]:
    """Normalize a source-path value (tuple/list/str/None) into a list of paths."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(p) for p in value if p]


@dataclass
class SharedWeightEndpoint:
    """One component-local endpoint of a shared Hugging Face parameter."""

    component: str
    parameter: str

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("Shared-weight endpoint component must be a non-empty string.")
        if not isinstance(self.parameter, str) or not self.parameter.endswith(".weight"):
            raise ValueError("Shared-weight endpoint parameter must name a '.weight' tensor.")

    @classmethod
    def coerce(cls, data: "SharedWeightEndpoint | dict | object") -> "SharedWeightEndpoint":
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(component=data["component"], parameter=data["parameter"])
        duck_data: Any = data
        return cls(
            component=duck_data.component,
            parameter=duck_data.parameter,
        )

    def to_json(self) -> dict[str, str]:
        return {"component": self.component, "parameter": self.parameter}


@dataclass
class SharedWeightInfo:
    """A logical parameter shared by components in the source Hugging Face model."""

    name: str
    canonical: SharedWeightEndpoint
    aliases: list[SharedWeightEndpoint] = field(default_factory=list)
    kind: str = "parameter_alias"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Shared-weight name must be a non-empty string.")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(f"Shared weight {self.name!r} must declare a kind.")
        if not self.aliases:
            raise ValueError(f"Shared weight {self.name!r} must declare at least one alias.")
        endpoints = self.endpoints
        names = [endpoint.parameter for endpoint in endpoints]
        if len(set(names)) != len(names):
            raise ValueError(f"Shared weight {self.name!r} contains duplicate parameter endpoints.")

    @classmethod
    def coerce(cls, data: "SharedWeightInfo | dict | object") -> "SharedWeightInfo":
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            return cls(
                name=data["name"],
                canonical=SharedWeightEndpoint.coerce(data["canonical"]),
                aliases=[SharedWeightEndpoint.coerce(alias) for alias in data.get("aliases", ())],
                kind=data.get("kind", "parameter_alias"),
            )
        duck_data: Any = data
        return cls(
            name=duck_data.name,
            canonical=SharedWeightEndpoint.coerce(duck_data.canonical),
            aliases=[SharedWeightEndpoint.coerce(alias) for alias in duck_data.aliases],
            kind=getattr(duck_data, "kind", "parameter_alias"),
        )

    @property
    def endpoints(self) -> list[SharedWeightEndpoint]:
        return [self.canonical, *self.aliases]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "canonical": self.canonical.to_json(),
            "aliases": [alias.to_json() for alias in self.aliases],
        }


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
        metadata: Additional component metadata retained from earlier callers.
        shared_weights: Cross-component shared-weight declarations from Mobius.

    """

    name: str
    role: Optional[str] = None
    source_paths: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    shared_weights: list[SharedWeightInfo] = field(default_factory=list)

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
            recognized = {
                "name",
                "role",
                "kind",
                "source",
                "source_path",
                "source_paths",
                "shared_weights",
            }
            return cls(
                name=data["name"],
                role=data.get("role") or data.get("kind"),
                source_paths=_as_path_list(source_paths),
                shared_weights=[
                    SharedWeightInfo.coerce(shared_weight) for shared_weight in data.get("shared_weights", ())
                ],
                metadata={k: v for k, v in data.items() if k not in recognized},
            )
        source_paths = getattr(data, "source_paths", None)
        if source_paths is None:
            source_paths = getattr(data, "source_path", None)
        duck_data: Any = data
        return cls(
            name=duck_data.name,
            role=getattr(duck_data, "role", None) or getattr(duck_data, "kind", None),
            source_paths=_as_path_list(source_paths),
            shared_weights=[
                SharedWeightInfo.coerce(shared_weight) for shared_weight in getattr(duck_data, "shared_weights", ())
            ],
            metadata=dict(getattr(duck_data, "metadata", {}) or {}),
        )


def inspect_components(
    model_name_or_path: str,
    task: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
) -> list[ComponentInfo]:
    """Return the component plan for a Hugging Face model by querying mobius.

    Args:
        model_name_or_path: Hugging Face model id or local path.
        task: Optional task hint passed to mobius.
        trust_remote_code: Whether to trust remote code when mobius loads the config. ``None``
            defers to the underlying library's default.

    Returns:
        A list of :class:`ComponentInfo`. An empty list means the model is single-component
        (no separable components).

    Raises:
        ImportError: If ``mobius-onnx`` is not installed.

    """
    try:
        import mobius
    except ImportError as exc:
        raise ImportError(
            "mobius-onnx is required to resolve model components for a Hugging Face model. "
            "Install with: pip install mobius-onnx"
        ) from exc

    raw_components = mobius.inspect_components(
        model_name_or_path,
        task=task,
        trust_remote_code=trust_remote_code,
    )
    components = [ComponentInfo.coerce(c) for c in raw_components]
    logger.debug("mobius.inspect_components(%s) -> %s", model_name_or_path, [c.name for c in components])
    return components
