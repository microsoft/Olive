# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from typing import Any, Callable, Optional, Union

from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config import IoConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler

logger = logging.getLogger(__name__)


@model_handler_registry("QairtPreparedModel")
class QairtPreparedModelHandler(OliveModelHandler):
    json_config_keys: tuple[str, ...] = ("io_config", "model_file_format")

    def __init__(
        self,
        model_path: str,
        model_attributes: Optional[dict[str, Any]] = None,
        io_config: Union[dict[str, Any], IoConfig, str, Callable] = None,
        model_file_format: ModelFileFormat = ModelFileFormat.QAIRT_PREPARED,
    ):
        super().__init__(
            framework=Framework.QAIRT,
            model_file_format=model_file_format,
            model_path=model_path,
            model_attributes=model_attributes,
            io_config=io_config,
        )

    @property
    def size_on_disk(self) -> int:
        """Compute size of the model on disk."""
        return 0

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError("QairtPreparedModelHandler does not support load_model")

    def prepare_session(
        self,
        inference_settings: Union[dict[str, Any], None] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, list[str]] = None,
        rank: Union[int, None] = None,
    ):
        raise NotImplementedError("QairtPreparedModelHandler does not support prepare_session")

    def run_session(
        self,
        session: Any = None,
        inputs: Union[dict[str, Any], list[Any], tuple[Any, ...]] = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        raise NotImplementedError("QairtPreparedModelHandler does not support prepare_session")


@model_handler_registry("QairtModel")
class QairtModelHandler(OliveModelHandler):
    json_config_keys: tuple[str, ...] = ("io_config", "model_file_format")

    def __init__(
        self,
        model_path: str,
        model_attributes: Optional[dict[str, Any]] = None,
        io_config: Union[dict[str, Any], IoConfig, str, Callable] = None,
        model_file_format: ModelFileFormat = ModelFileFormat.QAIRT,
    ):
        super().__init__(
            framework=Framework.QAIRT,
            model_file_format=model_file_format,
            model_path=model_path,
            model_attributes=model_attributes,
            io_config=io_config,
        )

    @property
    def size_on_disk(self) -> int:
        """Compute size of the model on disk."""
        return 0

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError("QairtModelHandler does not support load_model")

    def prepare_session(
        self,
        inference_settings: Union[dict[str, Any], None] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, list[str]] = None,
        rank: Union[int, None] = None,
    ):
        raise NotImplementedError("QairtModelHandler does not support prepare_session")

    def run_session(
        self,
        session: Any = None,
        inputs: Union[dict[str, Any], list[Any], tuple[Any, ...]] = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        raise NotImplementedError("QairtModelHandler does not support prepare_session")
