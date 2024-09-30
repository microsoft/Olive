# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import platform
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from olive.common.constants import OS
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config import IoConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.platform_sdk.qualcomm.qnn.qnn import QNNInferenceSession, QNNSessionOptions

logger = logging.getLogger(__name__)


@model_handler_registry("QNNModel")
class QNNModelHandler(OliveModelHandler):
    json_config_keys: Tuple[str, ...] = ("io_config", "model_file_format")

    def __init__(
        self,
        model_path: str,
        model_attributes: Optional[Dict[str, Any]] = None,
        io_config: Union[Dict[str, Any], IoConfig, str, Callable] = None,
        model_file_format: ModelFileFormat = ModelFileFormat.QNN_CPP,
    ):
        super().__init__(
            framework=Framework.QNN,
            model_file_format=model_file_format,
            model_path=model_path,
            model_attributes=model_attributes,
            io_config=io_config,
        )

    @property
    def model_path(self):
        model_path = super().model_path
        if self.model_file_format == ModelFileFormat.QNN_CPP:
            logger.debug("QNNModelHandler: model_path is the cpp file for QNN_CPP model format.")
        elif self.model_file_format == ModelFileFormat.QNN_LIB:
            # self.model_path is the folder containing the lib file, the structure is like:
            # - self.model_path
            #   - aarch64-android
            #     - libmodel.so
            #   - x86_64-linux-clang
            #     - libmodel.so
            model_attributes = self.model_attributes or {}
            model_lib_suffix = None
            lib_targets = model_attributes.get("lib_targets")
            if lib_targets is None:
                logger.debug(
                    "QNNModelHandler: lib_targets is not provided, using default lib_targets x86_64-linux-clang"
                )
                if platform.system() == OS.LINUX:
                    lib_targets = "x86_64-linux-clang"
                    model_lib_suffix = ".so"
                elif platform.system() == OS.WINDOWS:
                    # might be different for arm devices
                    lib_targets = "x64"
                    model_lib_suffix = ".dll"
            model_folder = Path(model_path) / lib_targets
            model_paths = list(model_folder.glob(f"*{model_lib_suffix}"))
            assert model_paths, f"No model file found in {model_folder}"
            assert len(model_paths) == 1, f"Multiple model files found in {model_folder}: {model_paths}"
            return str(model_paths[0])
        elif self.model_file_format == ModelFileFormat.QNN_SERIALIZED_BIN:
            logger.debug("QNNModelHandler: model_path is the .serialized.bin file for QNN_SERIALIZED_BIN model format.")
        else:
            raise ValueError(f"Unsupported model file format {self.model_file_format}")
        return model_path

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError("QNNModelHandler does not support load_model")

    def prepare_session(
        self,
        inference_settings: Union[Dict[str, Any], None] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Union[int, None] = None,
    ):
        inference_settings = inference_settings or {}
        model_attributes = self.model_attributes or {}
        inference_settings["model_file_format"] = inference_settings.get("model_file_format") or self.model_file_format
        # some model is build under specific backend, e.g. serialized bin model is built under HTP backend
        # in these cases, we should respect the backend specified in the model_attributes, then overwrite it with
        # the backend specified in inference_settings
        inference_settings["backend"] = model_attributes.get("backend") or inference_settings.get("backend")
        session_options = QNNSessionOptions(**inference_settings)
        return QNNInferenceSession(self.model_path, self.io_config, session_options)

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        return session(inputs, **kwargs)
