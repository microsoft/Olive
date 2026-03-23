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


@model_handler_registry("MultiTargetModel")
class MultiTargetModelHandler(OliveModelHandler):
    """MultiTargetModel represents the same model compiled for multiple hardware targets.

    Unlike CompositeModelHandler which holds different component models (e.g., split parts of a pipeline),
    MultiTargetModelHandler holds the same logical model compiled for different hardware targets
    (e.g., different SoC models for QNN).

    When a pass encounters a MultiTargetModelHandler, it runs independently on each target model,
    preserving the multi-target structure through the pipeline.
    """

    resource_keys: tuple[str, ...] = ("model_path",)
    json_config_keys: tuple[str, ...] = ("target_names",)

    def __init__(
        self,
        target_models: list[Union[OliveModelHandler, dict[str, Any]]],
        target_names: list[str],
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.COMPOSITE_MODEL,
            model_attributes=model_attributes,
        )
        self._target_models = [
            validate_config(m, ModelConfig).create_model() if isinstance(m, dict) else m for m in target_models
        ]
        assert all(isinstance(m, OliveModelHandler) for m in self._target_models), (
            "All target models must be OliveModelHandler or dict"
        )
        assert len(self._target_models) == len(target_names), "Number of target models and names must match"
        self.target_names = target_names

    @property
    def target_models(self):
        for m in self._target_models:
            m.model_attributes = {**(self.model_attributes or {}), **(m.model_attributes or {})}
            yield m

    def to_json(self, check_object: bool = False):
        json_dict = super().to_json(check_object)
        json_dict["config"]["target_models"] = []
        for m in self._target_models:
            target_json = m.to_json(check_object)
            target_json["config"]["model_attributes"] = dict_diff(
                target_json["config"]["model_attributes"], self.model_attributes
            )
            json_dict["config"]["target_models"].append(target_json)
        return serialize_to_json(json_dict, check_object)

    def get_target_models(self) -> list[tuple[str, OliveModelHandler]]:
        """Iterate over (target_name, target_model) pairs."""
        return zip(self.target_names, self.target_models)

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError

    @property
    def size_on_disk(self) -> int:
        """Compute size of the model on disk."""
        raise NotImplementedError

    def prepare_session(
        self,
        inference_settings: Optional[dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, list[str]] = None,
        rank: Optional[int] = None,
    ):
        raise RuntimeError("MultiTargetModelHandler doesn't have a session of its own")

    def run_session(
        self,
        session: Any = None,
        inputs: Union[dict[str, Any], list[Any], tuple[Any, ...]] = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        raise RuntimeError("MultiTargetModelHandler doesn't have a session of its own")
