# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_onnx_model

import pytest

from olive.model.config.model_config import ModelConfig
from olive.model.handler.composite import CompositeModelHandler
from olive.model.handler.onnx import ONNXModelHandler


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
