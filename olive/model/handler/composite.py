# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from olive.common.config_utils import serialize_to_json, validate_config
from olive.common.utils import dict_diff
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.model_config import ModelConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler

logger = logging.getLogger(__name__)


@model_handler_registry("CompositeModel")
class CompositeModelHandler(OliveModelHandler):
    """CompositeModel represents multiple component models.

    The only responsibility of CompositeModelHandler is to provider a get_model_components which will iterate all the
    child models.

    Whisper is an example composite model that has encoder and decoder components.
    CompositeModelHandler is a collection of Models. All the child model in the container should have same model type.
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
        self._model_components = [
            validate_config(m, ModelConfig).create_model() if isinstance(m, dict) else m for m in model_components
        ]
        assert all(
            isinstance(m, OliveModelHandler) for m in self._model_components
        ), "All components must be OliveModelHandler or dict"

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
        json_dict = {
            "type": self.model_type,
            "config": {"model_attributes": self.model_attributes, "model_component_names": self.model_component_names},
        }
        json_dict["config"]["model_components"] = []
        for m in self._model_components:
            component_json = m.to_json(check_object)
            # only keep attributes that are different from the parent
            component_json["config"]["model_attributes"] = dict_diff(
                component_json["config"]["model_attributes"], self.model_attributes
            )
            json_dict["config"]["model_components"].append(component_json)

        return serialize_to_json(json_dict, check_object)

    def get_model_components(self) -> List[Tuple[str, OliveModelHandler]]:
        return zip(self.model_component_names, self.model_components)

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        raise RuntimeError("CompositeModelHandler doesn't have a session of its own")

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        raise RuntimeError("CompositeModelHandler doesn't have a session of its own")
