# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import Mock

import pytest

from olive.common import mobius_utils
from olive.model.config.model_config import ModelConfig
from olive.model.handler.composite import CompositeModelHandler
from olive.model.handler.onnx import ONNXModelHandler
from test.utils import get_onnx_model


@pytest.mark.parametrize("as_handler", [True, False])
def test_composite_model(as_handler):
    # setup
    input_models = []
    child_attributes = {"attr0": "value0", "attr1": "value1"}
    parent_attributes = {"attr0": "value00", "attr1": "value1", "attr2": "value2"}
    input_models.append(get_onnx_model(model_attributes=child_attributes))
    input_models.append(get_onnx_model())
    if not as_handler:
        input_models = [model.to_json() for model in input_models]

    # create handler
    composite_model = CompositeModelHandler(
        input_models,
        ["component0", "component1"],
        model_attributes=parent_attributes,
    )

    # check components
    for component_name, model in composite_model.get_model_components():
        assert isinstance(model, ONNXModelHandler)
        if component_name == "component0":
            assert model.model_attributes == {**parent_attributes, **child_attributes}
        else:
            assert model.model_attributes == parent_attributes

    # check serialization
    composite_json = composite_model.to_json()
    # common model attributes are removed
    assert composite_json["config"]["model_components"][0]["config"]["model_attributes"] == {"attr0": "value0"}
    model_config = ModelConfig.from_json(composite_json)
    assert model_config.type == CompositeModelHandler.model_type


def test_composite_model_missing_names_raises_value_error():
    # model_components provided but model_component_names omitted: must raise a clear ValueError
    # (not a TypeError from len(None), which would also be silenced under `python -O` if asserted).
    with pytest.raises(ValueError, match="requires model_component_names"):
        CompositeModelHandler([get_onnx_model(), get_onnx_model()])


def test_composite_model_component_name_count_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="must match"):
        CompositeModelHandler([get_onnx_model(), get_onnx_model()], ["only_one_name"])


def _build_composite_handler():
    return CompositeModelHandler(
        [get_onnx_model(), get_onnx_model(), get_onnx_model()],
        ["text_encoder", "unet", "vae_decoder"],
        model_attributes={"shared": "value"},
    )


def test_select_components_single_returns_unwrapped_child():
    composite = _build_composite_handler()
    selected = composite.select_components(["unet"])
    assert isinstance(selected, ONNXModelHandler)
    # parent attributes should be inherited by the unwrapped child
    assert selected.model_attributes == {"shared": "value"}


def test_select_components_multiple_returns_subset_composite():
    composite = _build_composite_handler()
    selected = composite.select_components(["vae_decoder", "text_encoder"])
    assert isinstance(selected, CompositeModelHandler)
    # order from the call is preserved
    assert list(selected.model_component_names) == ["vae_decoder", "text_encoder"]


def test_select_components_unknown_name_raises():
    composite = _build_composite_handler()
    with pytest.raises(ValueError, match="Unknown component"):
        composite.select_components(["no_such_component"])


def test_select_components_empty_list_raises():
    composite = _build_composite_handler()
    with pytest.raises(ValueError, match="non-empty"):
        composite.select_components([])


def test_model_config_select_components_multiple_returns_composite_config():
    composite_config = ModelConfig.model_validate(
        {
            "type": "CompositeModel",
            "config": {
                "model_components": [
                    {"type": "ONNXModel", "config": {"model_path": "a.onnx"}},
                    {"type": "ONNXModel", "config": {"model_path": "b.onnx"}},
                    {"type": "ONNXModel", "config": {"model_path": "c.onnx"}},
                ],
                "model_component_names": ["text_encoder", "unet", "vae_decoder"],
            },
        }
    )
    selected = composite_config.select_components(["vae_decoder", "text_encoder"])
    assert isinstance(selected, ModelConfig)
    assert selected.type == "compositemodel"
    assert list(selected.config["model_component_names"]) == ["vae_decoder", "text_encoder"]
    assert [c["config"]["model_path"] for c in selected.config["model_components"]] == ["c.onnx", "a.onnx"]


def test_model_config_select_components_single_inherits_parent_attributes():
    composite_config = ModelConfig.model_validate(
        {
            "type": "CompositeModel",
            "config": {
                "model_components": [
                    {"type": "ONNXModel", "config": {"model_path": "a.onnx", "model_attributes": {"child": "c"}}},
                    {"type": "ONNXModel", "config": {"model_path": "b.onnx"}},
                ],
                "model_component_names": ["text_encoder", "unet"],
                "model_attributes": {"shared": "s", "child": "parent"},
            },
        }
    )
    selected = composite_config.select_components(["text_encoder"])
    assert isinstance(selected, ModelConfig)
    assert selected.type == "onnxmodel"
    # parent-only keys are inherited; child keys win on conflict
    assert selected.config["model_attributes"] == {"shared": "s", "child": "c"}


def test_model_config_get_components_returns_none_for_non_composite():
    onnx_config = ModelConfig.model_validate({"type": "ONNXModel", "config": {"model_path": "a.onnx"}})
    assert onnx_config.get_components() is None


def test_model_config_get_components_returns_names_for_composite():
    composite_config = ModelConfig.model_validate(
        {
            "type": "CompositeModel",
            "config": {
                "model_components": [
                    {"type": "ONNXModel", "config": {"model_path": "a.onnx"}},
                    {"type": "ONNXModel", "config": {"model_path": "b.onnx"}},
                ],
                "model_component_names": ["text_encoder", "unet"],
            },
        }
    )
    assert composite_config.get_components() == ["text_encoder", "unet"]


def _make_export_package(root):
    """Create an export package with one model.onnx subfolder per component."""
    for name in ["decoder", "vision_encoder", "embedding"]:
        comp_dir = root / name
        comp_dir.mkdir(parents=True)
        (comp_dir / "model.onnx").write_bytes(b"onnx")
    return root


def test_composite_handler_discovers_components_from_directory(tmp_path):
    _make_export_package(tmp_path)
    handler = CompositeModelHandler(model_path=str(tmp_path))
    assert list(handler.model_component_names) == ["decoder", "embedding", "vision_encoder"]
    for _, component in handler.get_model_components():
        assert isinstance(component, ONNXModelHandler)


def test_model_config_get_components_discovers_directory_composite(tmp_path):
    _make_export_package(tmp_path)
    config = ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(tmp_path)}})
    assert config.get_components() == ["decoder", "embedding", "vision_encoder"]


def test_model_config_get_components_hfmodel_uses_mobius(monkeypatch):
    inspect_components = Mock(
        return_value=[mobius_utils.ComponentInfo(name="decoder", role="decoder", source_paths=["model.language_model"])]
    )
    monkeypatch.setattr(mobius_utils, "inspect_components", inspect_components)
    config = ModelConfig.model_validate(
        {
            "type": "HfModel",
            "config": {
                "model_path": "some/vlm",
                "load_kwargs": {"trust_remote_code": True},
                "model_attributes": {"mobius_task": "qwen-vl"},
            },
        }
    )

    assert config.get_components() == ["decoder"]
    inspect_components.assert_called_once_with("some/vlm", task="qwen-vl", trust_remote_code=True)


def test_model_config_select_components_hfmodel_tags_component(monkeypatch):
    monkeypatch.setattr(
        mobius_utils,
        "inspect_components",
        lambda *args, **kwargs: [
            mobius_utils.ComponentInfo(name="decoder", role="decoder", source_paths=["model.language_model"])
        ],
    )
    config = ModelConfig.model_validate({"type": "HfModel", "config": {"model_path": "some/vlm"}})

    selected = config.select_components(["decoder"])

    assert selected.type == "hfmodel"
    assert selected.config["model_path"] == "some/vlm"
    assert selected.config["model_attributes"] == {
        "component_name": "decoder",
        "component_role": "decoder",
        "component_source_paths": ["model.language_model"],
    }


def test_model_config_select_components_hfmodel_multiple_names_raises():
    config = ModelConfig.model_validate({"type": "HfModel", "config": {"model_path": "some/vlm"}})

    with pytest.raises(ValueError, match="one at a time"):
        config.select_components(["decoder", "vision_encoder"])


def test_model_config_select_components_hfmodel_missing_paths_raises(monkeypatch):
    monkeypatch.setattr(
        mobius_utils,
        "inspect_components",
        lambda *args, **kwargs: [
            mobius_utils.ComponentInfo(name="decoder", role="decoder"),
            mobius_utils.ComponentInfo(name="vision_encoder", role="encoder", source_paths=["model.visual"]),
        ],
    )
    config = ModelConfig.model_validate({"type": "HfModel", "config": {"model_path": "some/vlm"}})

    with pytest.raises(ValueError, match="no runtime source paths"):
        config.select_components(["decoder"])


def test_model_config_select_components_hfmodel_whole_model_allows_empty_paths(monkeypatch):
    monkeypatch.setattr(
        mobius_utils,
        "inspect_components",
        lambda *args, **kwargs: [mobius_utils.ComponentInfo(name="model", role="decoder")],
    )
    config = ModelConfig.model_validate({"type": "HfModel", "config": {"model_path": "some/llm"}})

    selected = config.select_components(["model"])

    assert selected.config["model_attributes"] == {"component_name": "model", "component_role": "decoder"}


def _make_diffusers_dir(tmp_path):
    """Create a minimal local diffusers dir so is_valid_diffusers_model passes offline."""
    (tmp_path / "model_index.json").write_text("{}")
    return tmp_path


def test_model_config_select_components_diffusersmodel_scopes_subset(tmp_path):
    model_dir = _make_diffusers_dir(tmp_path)
    config = ModelConfig.model_validate(
        {"type": "DiffusersModel", "config": {"model_path": str(model_dir), "model_variant": "sdxl"}}
    )
    selected = config.select_components(["unet", "text_encoder"])
    assert selected.type == "diffusersmodel"
    # preserved in the variant's canonical order, not the requested order
    assert selected.config["components"] == ["text_encoder", "unet"]
    # the scoped config now exposes only the selected components
    assert selected.get_components() == ["text_encoder", "unet"]


def test_model_config_select_components_diffusersmodel_unknown_raises(tmp_path):
    model_dir = _make_diffusers_dir(tmp_path)
    config = ModelConfig.model_validate(
        {"type": "DiffusersModel", "config": {"model_path": str(model_dir), "model_variant": "sd"}}
    )
    with pytest.raises(ValueError, match="Unknown component name"):
        config.select_components(["text_encoder_2"])  # SDXL-only; not in SD
