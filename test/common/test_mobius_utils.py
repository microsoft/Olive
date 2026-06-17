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
    kind: str


def test_coerce_returns_same_instance_when_already_componentinfo():
    component = ComponentInfo(name="decoder", kind="decoder", source_path="model.language_model")

    assert ComponentInfo.coerce(component) is component


def test_coerce_reads_contract_dict():
    component = ComponentInfo.coerce(
        {"name": "decoder", "kind": "decoder", "source": {"path": "model.language_model"}, "extra": 1}
    )

    assert component.name == "decoder"
    assert component.kind == "decoder"
    assert component.source_path == "model.language_model"
    assert component.metadata == {"extra": 1}


def test_coerce_reads_duck_typed_object_when_object_has_no_mapping_interface():
    # A mobius ComponentInfo is a frozen dataclass and does not implement ``.get``.
    component = ComponentInfo.coerce(_MobiusLikeComponent(name="vision_encoder", kind="encoder"))

    assert component.name == "vision_encoder"
    assert component.kind == "encoder"
    assert component.source_path is None


def test_inspect_components_coerces_mobius_objects(monkeypatch):
    fake_mobius = types.ModuleType("mobius")
    fake_mobius.inspect_components = lambda model_name_or_path, task=None, trust_remote_code=False: [
        _MobiusLikeComponent(name="decoder", kind="decoder"),
        _MobiusLikeComponent(name="vision_encoder", kind="encoder"),
        _MobiusLikeComponent(name="embedding", kind="embedding"),
    ]
    monkeypatch.setitem(sys.modules, "mobius", fake_mobius)

    components = inspect_components("fake/llava")

    assert all(isinstance(c, ComponentInfo) for c in components)
    assert [(c.name, c.kind, c.source_path) for c in components] == [
        ("decoder", "decoder", None),
        ("vision_encoder", "encoder", None),
        ("embedding", "embedding", None),
    ]


def test_inspect_components_raises_importerror_when_mobius_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "mobius", None)

    with pytest.raises(ImportError, match="mobius-ai is required"):
        inspect_components("fake/llava")
