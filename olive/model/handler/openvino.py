# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, create_resource_path

if TYPE_CHECKING:
    try:
        from openvino import Model
    except ImportError:
        raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None


@model_handler_registry("OpenVINOModel")
class OpenVINOModelHandler(OliveModelHandler):
    """OpenVINO model handler.

    The main responsibility of OpenVINOModelHandler is to provide the model loading for OpenVINO model.
    """

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model: Optional["Model"] = None,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.OPENVINO,
            model_file_format=ModelFileFormat.OPENVINO_IR,
            model=model,
            model_attributes=model_attributes,
        )
        # check if the model files (xml, bin) are in the same directory
        if model_path is not None:
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

    def load_model(self, rank: int = None, enable_fast_mode: bool = False):
        if self.model is not None:
            return self.model

        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None
        self.model = ov.Core().read_model(self.model_config["model"])
        return self.model

    def save_model_to_path(self, save_path: Union[str, Path], model_name: Optional[str] = None):
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        if self.model is None:
            raise ValueError("Model is not loaded yet. Cannot save model to file.")
        model_name = "ov_model"
        output_dir = Path(save_path) / model_name
        ov.save_model(self.model, output_model=output_dir.with_suffix(".xml"))
        return OpenVINOModelHandler(model_path=save_path)

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
