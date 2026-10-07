# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
import types
from unittest.mock import Mock

import pytest

from olive.common.mobius_utils import ComponentInfo, SharedWeightInfo, inspect_components


def test_coerce_reads_contract_dict():
    component = ComponentInfo.coerce(
        {"name": "decoder", "role": "decoder", "source": {"path": "model.language_model"}, "extra": 1}
    )

    assert component.name == "decoder"
    assert component.role == "decoder"
    assert component.source_paths == ["model.language_model"]
    assert component.metadata == {"extra": 1}


def test_component_info_keeps_metadata_as_fourth_positional_argument():
    metadata = {"source": "legacy"}

    component = ComponentInfo("decoder", "decoder", ["model.layers"], metadata)

    assert component.metadata is metadata
    assert not component.shared_weights


def test_coerce_reads_mobius_source_paths_tuple():
    # A component may span multiple disjoint HF sub-trees (e.g. phi4mm decoder).
    component = ComponentInfo.coerce(
        types.SimpleNamespace(
            name="decoder",
            role="decoder",
            source_paths=("model.layers", "model.norm", "lm_head"),
        )
    )

    assert component.role == "decoder"
    assert component.source_paths == ["model.layers", "model.norm", "lm_head"]


def test_coerce_reads_cross_component_shared_weights():
    component = ComponentInfo.coerce(
        types.SimpleNamespace(
            name="decoder",
            role="decoder",
            source_paths=("model.layers", "lm_head"),
            shared_weights=(
                types.SimpleNamespace(
                    name="word_embeddings",
                    kind="tied_word_embeddings",
                    canonical=types.SimpleNamespace(
                        component="embedding",
                        parameter="model.embed_tokens.weight",
                    ),
                    aliases=(
                        types.SimpleNamespace(
                            component="decoder",
                            parameter="lm_head.weight",
                        ),
                    ),
                ),
            ),
        )
    )

    assert component.shared_weights == [
        SharedWeightInfo.coerce(
            {
                "name": "word_embeddings",
                "kind": "tied_word_embeddings",
                "canonical": {
                    "component": "embedding",
                    "parameter": "model.embed_tokens.weight",
                },
                "aliases": [
                    {
                        "component": "decoder",
                        "parameter": "lm_head.weight",
                    }
                ],
            }
        )
    ]


@pytest.mark.parametrize(
    ("aliases", "message"),
    [
        ([], "at least one alias"),
        ([{"component": "decoder", "parameter": "model.embed_tokens.weight"}], "duplicate parameter"),
        (
            [
                {"component": "decoder", "parameter": "lm_head.weight"},
                {"component": "decoder", "parameter": "lm_head.weight"},
            ],
            "duplicate parameter",
        ),
    ],
)
def test_coerce_rejects_invalid_shared_weight_endpoints(aliases, message):
    with pytest.raises(ValueError, match=message):
        SharedWeightInfo.coerce(
            {
                "name": "word_embeddings",
                "canonical": {
                    "component": "embedding",
                    "parameter": "model.embed_tokens.weight",
                },
                "aliases": aliases,
            }
        )


def test_coerce_falls_back_to_legacy_kind_and_source_path():
    # Older mobius releases expose ``kind``/``source_path`` (singular string).
    component = ComponentInfo.coerce(
        types.SimpleNamespace(name="decoder", kind="decoder", source_path="model.language_model")
    )

    assert component.role == "decoder"
    assert component.source_paths == ["model.language_model"]


def test_inspect_components_coerces_mobius_objects(monkeypatch):
    fake_mobius = types.ModuleType("mobius")
    fake_mobius.inspect_components = Mock(
        return_value=[
            types.SimpleNamespace(name="decoder", role="decoder", source_paths=("model.language_model",)),
            types.SimpleNamespace(name="vision_encoder", role="encoder", source_paths=("model.visual",)),
            types.SimpleNamespace(name="embedding", role="embedding"),
        ]
    )
    monkeypatch.setitem(sys.modules, "mobius", fake_mobius)

    components = inspect_components("fake/llava")

    fake_mobius.inspect_components.assert_called_once_with("fake/llava", task=None, trust_remote_code=None)
    assert all(isinstance(c, ComponentInfo) for c in components)
    assert [(c.name, c.role, c.source_paths) for c in components] == [
        ("decoder", "decoder", ["model.language_model"]),
        ("vision_encoder", "encoder", ["model.visual"]),
        ("embedding", "embedding", []),
    ]


def test_inspect_components_raises_importerror_when_mobius_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "mobius", None)

    with pytest.raises(ImportError, match="mobius-onnx is required"):
        inspect_components("fake/llava")
