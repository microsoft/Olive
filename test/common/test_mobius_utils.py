# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import dataclasses
import sys
import types

import pytest

from olive.common.mobius_utils import ComponentInfo, inspect_components


@dataclasses.dataclass(frozen=True)
class _MobiusLikeComponent:
    """Mirror of mobius' frozen ``ComponentInfo`` dataclass (no mapping interface)."""

    name: str
    role: str
    source_paths: tuple = ()


@dataclasses.dataclass(frozen=True)
class _LegacyMobiusComponent:
    """Mirror of an older mobius ``ComponentInfo`` (pre-rename ``kind``/``source_path``)."""

    name: str
    kind: str
    source_path: str = None


def test_coerce_returns_same_instance_when_already_componentinfo():
    component = ComponentInfo(name="decoder", role="decoder", source_paths=["model.language_model"])

    assert ComponentInfo.coerce(component) is component


def test_coerce_reads_contract_dict():
    component = ComponentInfo.coerce(
        {"name": "decoder", "role": "decoder", "source": {"path": "model.language_model"}, "extra": 1}
    )

    assert component.name == "decoder"
    assert component.role == "decoder"
    assert component.source_paths == ["model.language_model"]
    assert component.metadata == {"extra": 1}


def test_coerce_reads_duck_typed_object_when_object_has_no_mapping_interface():
    # A mobius ComponentInfo is a frozen dataclass and does not implement ``.get``.
    component = ComponentInfo.coerce(_MobiusLikeComponent(name="vision_encoder", role="encoder"))

    assert component.name == "vision_encoder"
    assert component.role == "encoder"
    assert component.source_paths == []


def test_coerce_reads_mobius_source_paths_tuple():
    # A component may span multiple disjoint HF sub-trees (e.g. phi4mm decoder).
    component = ComponentInfo.coerce(
        _MobiusLikeComponent(
            name="decoder",
            role="decoder",
            source_paths=("model.layers", "model.norm", "lm_head"),
        )
    )

    assert component.role == "decoder"
    assert component.source_paths == ["model.layers", "model.norm", "lm_head"]


def test_coerce_falls_back_to_legacy_kind_and_source_path():
    # Older mobius releases expose ``kind``/``source_path`` (singular string).
    component = ComponentInfo.coerce(
        _LegacyMobiusComponent(name="decoder", kind="decoder", source_path="model.language_model")
    )

    assert component.role == "decoder"
    assert component.source_paths == ["model.language_model"]


def test_inspect_components_coerces_mobius_objects(monkeypatch):
    fake_mobius = types.ModuleType("mobius")
    fake_mobius.inspect_components = lambda model_name_or_path, task=None, trust_remote_code=False: [
        _MobiusLikeComponent(name="decoder", role="decoder", source_paths=("model.language_model",)),
        _MobiusLikeComponent(name="vision_encoder", role="encoder", source_paths=("model.visual",)),
        _MobiusLikeComponent(name="embedding", role="embedding"),
    ]
    monkeypatch.setitem(sys.modules, "mobius", fake_mobius)

    components = inspect_components("fake/llava")

    assert all(isinstance(c, ComponentInfo) for c in components)
    assert [(c.name, c.role, c.source_paths) for c in components] == [
        ("decoder", "decoder", ["model.language_model"]),
        ("vision_encoder", "encoder", ["model.visual"]),
        ("embedding", "embedding", []),
    ]


def test_inspect_components_raises_importerror_when_mobius_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "mobius", None)

    with pytest.raises(ImportError, match="mobius-ai is required"):
        inspect_components("fake/llava")
