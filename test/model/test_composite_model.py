# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

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


def test_model_config_select_components_single_returns_child_config():
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
    selected = composite_config.select_components(["unet"])
    assert isinstance(selected, ModelConfig)
    assert selected.type == "onnxmodel"
    assert selected.config["model_path"] == "b.onnx"


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


def test_model_config_select_components_on_non_composite_raises():
    onnx_config = ModelConfig.model_validate({"type": "ONNXModel", "config": {"model_path": "a.onnx"}})
    with pytest.raises(ValueError, match="only supported on CompositeModel"):
        onnx_config.select_components(["any"])


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
    """Create a mobius-style export package: one subfolder per component with a model.onnx."""
    for name in ["decoder", "vision_encoder", "embedding"]:
        comp_dir = root / name
        comp_dir.mkdir(parents=True)
        (comp_dir / "model.onnx").write_bytes(b"onnx")
    return root


def test_discover_onnx_components_reads_subfolders(tmp_path):
    from olive.model.utils.onnx_utils import discover_onnx_components

    _make_export_package(tmp_path)
    discovered = discover_onnx_components(str(tmp_path))
    assert [name for name, _ in discovered] == ["decoder", "embedding", "vision_encoder"]
    assert dict(discovered)["decoder"] == "decoder/model.onnx"


def test_discover_onnx_components_empty_for_flat_dir(tmp_path):
    from olive.model.utils.onnx_utils import discover_onnx_components

    (tmp_path / "model.onnx").write_bytes(b"onnx")
    assert not discover_onnx_components(str(tmp_path))


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


def test_model_config_select_components_discovers_directory_composite(tmp_path):
    _make_export_package(tmp_path)
    config = ModelConfig.model_validate({"type": "CompositeModel", "config": {"model_path": str(tmp_path)}})
    selected = config.select_components(["decoder"])
    assert isinstance(selected, ModelConfig)
    assert selected.type == "onnxmodel"
    assert selected.config["onnx_file_name"] == "decoder/model.onnx"


def test_model_config_get_components_hfmodel_queries_mobius(monkeypatch):
    from olive.common import mobius_utils

    monkeypatch.setattr(
        mobius_utils,
        "inspect_components",
        lambda *a, **k: [
            mobius_utils.ComponentInfo(name="decoder", kind="decoder", source_path="model.language_model"),
            mobius_utils.ComponentInfo(name="vision_encoder", kind="vision_encoder", source_path="model.vision_tower"),
        ],
    )
    config = ModelConfig.model_validate({"type": "HfModel", "config": {"model_path": "some/vlm"}})
    assert config.get_components() == ["decoder", "vision_encoder"]


def test_model_config_select_components_hfmodel_tags_source_path(monkeypatch):
    from olive.common import mobius_utils

    monkeypatch.setattr(
        mobius_utils,
        "inspect_components",
        lambda *a, **k: [
            mobius_utils.ComponentInfo(name="decoder", kind="decoder", source_path="model.language_model"),
        ],
    )
    config = ModelConfig.model_validate({"type": "HfModel", "config": {"model_path": "some/vlm"}})
    selected = config.select_components(["decoder"])
    assert selected.type == "hfmodel"
    assert selected.config["model_path"] == "some/vlm"
    attrs = selected.config["model_attributes"]
    assert attrs["component_name"] == "decoder"
    assert attrs["component_kind"] == "decoder"
    assert attrs["component_source_path"] == "model.language_model"


def test_model_config_select_components_hfmodel_multiple_names_raises(monkeypatch):
    from olive.common import mobius_utils

    monkeypatch.setattr(mobius_utils, "inspect_components", lambda *a, **k: [])
    config = ModelConfig.model_validate({"type": "HfModel", "config": {"model_path": "some/vlm"}})
    with pytest.raises(ValueError, match="one at a time"):
        config.select_components(["decoder", "vision_encoder"])


def _make_diffusers_dir(tmp_path):
    """Create a minimal local diffusers dir so is_valid_diffusers_model passes offline."""
    (tmp_path / "model_index.json").write_text("{}")
    return tmp_path


def test_model_config_get_components_diffusersmodel(tmp_path):
    model_dir = _make_diffusers_dir(tmp_path)
    config = ModelConfig.model_validate(
        {"type": "DiffusersModel", "config": {"model_path": str(model_dir), "model_variant": "sdxl"}}
    )
    assert config.get_components() == [
        "text_encoder",
        "text_encoder_2",
        "unet",
        "vae_encoder",
        "vae_decoder",
    ]


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
