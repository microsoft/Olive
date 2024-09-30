# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List, Optional, Tuple, Union

from olive.common.config_utils import serialize_to_json
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.platform_sdk.qualcomm.constants import SNPEDevice
from olive.platform_sdk.qualcomm.snpe import SNPEInferenceSession, SNPESessionOptions
from olive.platform_sdk.qualcomm.snpe.tools.dev import get_dlc_metrics
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS


@model_handler_registry("SNPEModel")
class SNPEModelHandler(OliveModelHandler):
    def __init__(
        self,
        input_names: List[str],
        input_shapes: List[List[int]],
        output_names: List[str],
        output_shapes: List[List[int]],
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            framework=Framework.SNPE,
            model_file_format=ModelFileFormat.SNPE_DLC,
            model_path=model_path,
            model_attributes=model_attributes,
            io_config={
                "input_names": input_names,
                "input_shapes": input_shapes,
                "output_names": output_names,
                "output_shapes": output_shapes,
            },
        )

    @property
    def io_config(self) -> Dict[str, Any]:
        assert self._io_config, "SNPEModelHandler: io_config is not set"

        keys = {"input_names", "input_shapes", "output_names", "output_shapes"}
        return {k: v for k, v in self._io_config.items() if k in keys}

    def load_model(self, rank: int = None, cache_model: bool = True):
        raise NotImplementedError

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ) -> SNPEInferenceSession:
        inference_settings = inference_settings or {}
        session_options = SNPESessionOptions(**inference_settings)
        if device == Device.NPU:
            device = SNPEDevice.DSP
        session_options.device = device
        return SNPEInferenceSession(self.model_path, self.io_config, session_options)

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        return session(inputs, **kwargs)

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update(self.io_config)
        return serialize_to_json(config, check_object)

    def get_dlc_metrics(self) -> dict:
        return get_dlc_metrics(self.model_path)
