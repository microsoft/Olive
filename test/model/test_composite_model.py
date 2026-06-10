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
