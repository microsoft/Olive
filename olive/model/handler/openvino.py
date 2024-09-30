# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, create_resource_path


@model_handler_registry("OpenVINOModel")
class OpenVINOModelHandler(OliveModelHandler):
    """OpenVINO model handler.

    The main responsibility of OpenVINOModelHandler is to provide the model loading for OpenVINO model.
    """

    def __init__(self, model_path: OLIVE_RESOURCE_ANNOTATIONS, model_attributes: Optional[Dict[str, Any]] = None):
        super().__init__(
            model_path=model_path,
            framework=Framework.OPENVINO,
            model_file_format=ModelFileFormat.OPENVINO_IR,
            model_attributes=model_attributes,
        )
        # check if the model files (xml, bin) are in the same directory
        model_path = create_resource_path(self.model_path)
        assert model_path.is_local_resource(), "OpenVINO model_path must be local file or directory."
        _ = self.model_config

    @property
    def model_config(self) -> Dict[str, str]:
        """Get the model configuration for OpenVINO model."""
        model_path = self.model_path
        assert Path(model_path).is_dir(), f"OpenVINO model path {model_path} is not a directory"

        if len(list(Path(model_path).glob("*.xml"))) == 0 or len(list(Path(model_path).glob("*.bin"))) == 0:
            raise FileNotFoundError(f"No OpenVINO model found in {model_path}")
        if len(list(Path(model_path).glob("*.xml"))) > 1 or len(list(Path(model_path).glob("*.bin"))) > 1:
            raise FileExistsError(f"More than 1 OpenVINO models are found in {model_path}")

        for model_file in Path(model_path).glob("*.xml"):
            ov_model = Path(model_file)
        for weights_file in Path(model_path).glob("*.bin"):
            ov_weights = Path(weights_file)

        return {
            "model_name": ov_model.stem,
            "model": str(ov_model.resolve()),
            "weights": str(ov_weights.resolve()),
        }

    def load_model(self, rank: int = None, cache_model: bool = True):
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None
        core = ov.Core()
        return core.read_model(self.model_config["model"])

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None
        core = ov.Core()
        if inference_settings and inference_settings.get("device_name"):
            device = inference_settings["device_name"]
        elif device == Device.INTEL_MYRIAD:
            device = "MYRIAD"
        compiled_model = core.compile_model(self.model_config["model"], device.upper())
        return compiled_model.create_infer_request()

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        return session.infer(inputs, **kwargs)
