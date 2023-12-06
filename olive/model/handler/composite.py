# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

from olive.common.config_utils import serialize_to_json, validate_config
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.model_config import ModelConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler


@model_handler_registry("CompositeModel")
class CompositeModelHandler(OliveModelHandler):
    """CompositeModel represents multiple component models.

    Whisper is an example composite model that has encoder and decoder components.
    CompositeOnnxModel is a collection of OnnxModels.
    """

    def __init__(
        self,
        model_components: List[Union[OliveModelHandler, Dict[str, Any]]],
        model_component_names: List[str],
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=None,
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.COMPOSITE_MODEL,
            model_attributes=model_attributes,
        )
        if isinstance(model_components[0], dict):
            self.model_components = [validate_config(m, ModelConfig).create_model() for m in model_components]
        else:
            assert all(
                isinstance(m, OliveModelHandler) for m in model_components
            ), "All components must be OliveModelHandler"
            self.model_components = model_components

        assert len(self.model_components) == len(model_component_names), "Number of components and names must match"
        self.model_component_names = model_component_names
        for m in self.model_components:
            m.set_composite_parent(self)

    def to_json(self, check_object: bool = False):
        json_dict = {
            "type": self.__class__.__name__,
            "config": {"model_attributes": self.model_attributes, "model_component_names": self.model_component_names},
        }
        json_dict["config"]["model_components"] = []
        for m in self.model_components:
            json_dict["config"]["model_components"].append(m.to_json(check_object))

        return serialize_to_json(json_dict, check_object)

    def get_model_components(self):
        return zip(self.model_component_names, self.model_components)

    def load_model(self, rank: int = None):
        raise NotImplementedError

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        raise NotImplementedError


@model_handler_registry("CompositePyTorchModel")
class CompositePyTorchModelHandler(CompositeModelHandler):
    def __init__(self, model_components: List[Dict[str, Any]], **kwargs):
        kwargs = kwargs or {}
        kwargs["model_file_format"] = ModelFileFormat.COMPOSITE_MODEL

        model_names = []
        pytorch_models = []
        for model_config in model_components:
            config_copy = deepcopy(model_config)

            assert "name" in config_copy
            model_name = config_copy["name"]
            del config_copy["name"]

            model_names.append(model_name)
            pytorch_models.append(validate_config(config_copy, ModelConfig).create_model())

        kwargs["model_components"] = pytorch_models
        kwargs["model_component_names"] = model_names
        super().__init__(**kwargs)
