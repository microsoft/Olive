# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Optional, Union

from olive.common.config_utils import serialize_to_json, validate_config
from olive.common.utils import dict_diff
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.model_config import ModelConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)


@model_handler_registry("CompositeModel")
class CompositeModelHandler(OliveModelHandler):
    """CompositeModel represents multiple component models.

    The only responsibility of CompositeModelHandler is to provider a get_model_components which will iterate all the
    child models.

    CompositeModelHandler is a collection of Models. All the child model in the container should have same model type.
    """

    resource_keys: tuple[str, ...] = ("model_path",)
    json_config_keys: tuple[str, ...] = ("model_component_names",)

    def __init__(
        self,
        model_components: list[Union[OliveModelHandler, dict[str, Any]]],
        model_component_names: list[str],
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.COMPOSITE_MODEL,
            model_attributes=model_attributes,
        )
        self._model_components = [
            validate_config(m, ModelConfig).create_model() if isinstance(m, dict) else m for m in model_components
        ]
        assert all(isinstance(m, OliveModelHandler) for m in self._model_components), (
            "All components must be OliveModelHandler or dict"
        )

        assert len(self._model_components) == len(model_component_names), "Number of components and names must match"
        self.model_component_names = model_component_names

    @property
    def model_components(self):
        for m in self._model_components:
            # the parent attributes should be inherited by the child model
            # child attributes take precedence
            m.model_attributes = {**(self.model_attributes or {}), **(m.model_attributes or {})}
            yield m

    def to_json(self, check_object: bool = False):
        json_dict = super().to_json(check_object)
        json_dict["config"]["model_components"] = []
        for m in self._model_components:
            component_json = m.to_json(check_object)
            # only keep attributes that are different from the parent
            component_json["config"]["model_attributes"] = dict_diff(
                component_json["config"]["model_attributes"], self.model_attributes
            )
            json_dict["config"]["model_components"].append(component_json)

        return serialize_to_json(json_dict, check_object)

    def get_model_components(self) -> list[tuple[str, OliveModelHandler]]:
        return zip(self.model_component_names, self.model_components)

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError

    def prepare_session(
        self,
        inference_settings: Optional[dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, list[str]] = None,
        rank: Optional[int] = None,
    ):
        raise RuntimeError("CompositeModelHandler doesn't have a session of its own")

    def run_session(
        self,
        session: Any = None,
        inputs: Union[dict[str, Any], list[Any], tuple[Any, ...]] = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        raise RuntimeError("CompositeModelHandler doesn't have a session of its own")
