# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List, Optional, Tuple, Union

from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS


@model_handler_registry("TensorFlowModel")
class TensorFlowModelHandler(OliveModelHandler):
    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_file_format: ModelFileFormat = ModelFileFormat.TENSORFLOW_SAVED_MODEL,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.TENSORFLOW,
            model_file_format=model_file_format,
            model_attributes=model_attributes,
        )

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        raise NotImplementedError

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        raise NotImplementedError
