# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import onnx
from onnx import GraphProto, ModelProto

from olive.common.config_utils import serialize_to_json
from olive.common.ort_inference import get_ort_inference_session
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import AcceleratorLookup, Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.model.mixin import OnnxEpValidateMixin, OnnxGraphMixin
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)


@model_handler_registry("ONNXModel")
class ONNXModelHandler(OliveModelHandler, OnnxEpValidateMixin, OnnxGraphMixin):
    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        onnx_file_name: Optional[str] = None,
        inference_settings: Optional[dict] = None,
        use_ort_extensions: bool = False,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.ONNX,
            model_path=model_path,
            model_attributes=model_attributes,
        )
        self.inference_settings = inference_settings
        self.use_ort_extensions = use_ort_extensions
        self.onnx_file_name = onnx_file_name

        self.io_config = None
        self.graph = None
        self.all_graphs: Optional[List[GraphProto]] = None

        # check for onnx file name since it will do validation
        _ = self.model_path

    @property
    def model_path(self) -> str:
        model_path = super().model_path
        return get_onnx_file_path(model_path, self.onnx_file_name) if model_path else None

    def load_model(self, rank: int = None) -> ModelProto:
        return onnx.load(self.model_path)

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        # user provided inference_settings > model's inference_settings > default settings
        inference_settings = inference_settings or self.inference_settings or {}
        # deep copy to avoid modifying the original settings
        inference_settings = deepcopy(inference_settings)

        # if user doesn't not provide ep list, use default value([ep]). Otherwise, use the user's ep list
        # user provided ep list > eps given by arguments > default eps
        execution_providers = inference_settings.get("execution_provider") or execution_providers
        if not execution_providers:
            execution_providers = self.get_default_execution_providers(device)
        elif isinstance(execution_providers, str):
            execution_providers = [execution_providers]
        else:
            # the execution_providers is a list
            pass
        inference_settings["execution_provider"] = execution_providers

        if (device == Device.GPU) and (rank is not None) and not inference_settings.get("provider_options"):
            inference_settings["provider_options"] = [
                {"device_id": str(rank)} if ep == "CUDAExecutionProvider" else {} for ep in execution_providers
            ]

        return get_ort_inference_session(self.model_path, inference_settings, self.use_ort_extensions)

    def get_default_execution_providers(self, device: Device):
        # return firstly available ep as ort default ep
        available_providers = AcceleratorLookup.get_execution_providers_for_device(device)
        for ep in available_providers:
            if self.is_valid_ep(self.model_path, ep):
                return [ep]
        return ["CPUExecutionProvider"]

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update(
            {
                "onnx_file_name": self.onnx_file_name,
                "inference_settings": self.inference_settings,
                "use_ort_extensions": self.use_ort_extensions,
            }
        )
        return serialize_to_json(config, check_object)

    def get_io_config(self):
        """Get input/output names, shapes, types of the onnx model without creating an ort session.

        This function loads the onnx model and parses the graph to get the io config.
        """
        if self.io_config:
            return self.io_config

        # save io_config
        self.io_config = self.get_graph_io_config()
        return self.io_config


def resolve_path(file_or_dir_path: str, model_filename: str = "model.onnx") -> str:
    """Get the model full path.

    The engine provides output paths to ONNX passes that do not contain .onnx extension
    (these paths are generally locations in the cache). This function will convert such
    paths to absolute file paths and also ensure the parent directories exist.
    If the input path is already an ONNX file it is simply returned. Examples:

    resolve_path("c:/foo/bar.onnx") -> c:/foo/bar.onnx

    resolve_path("c:/foo/bar") -> c:/foo/bar/model.onnx
    """
    if not model_filename.endswith(".onnx"):
        raise ValueError(f"ONNXModel's model name must end with '.onnx', got {model_filename}")

    path = Path(file_or_dir_path)
    if path.suffix != ".onnx":
        path = path / model_filename
        parent_dir = path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_onnx_file_path(model_path: str, onnx_file_name: Optional[str] = None) -> str:
    """Get the path to the ONNX model file.

    If model_path is a file, it is returned as is. If model_path is a
    directory, the onnx_file_name is appended to it and the resulting path is returned. If onnx_file_name is not
    specified, it is inferred if there is only one .onnx file in the directory, else an error is raised.
    """
    assert Path(model_path).exists(), f"Model path {model_path} does not exist"

    # if model_path is a file, return it as is
    if Path(model_path).is_file():
        return model_path

    # if model_path is a directory, append onnx_file_name to it
    if onnx_file_name:
        onnx_file_path = Path(model_path) / onnx_file_name
        assert onnx_file_path.exists(), f"ONNX model file {onnx_file_path} does not exist"
        return str(onnx_file_path)

    # try to infer onnx_file_name
    logger.warning(
        "model_path is a directory but onnx_file_name is not specified. Trying to infer it. It is recommended to"
        " specify onnx_file_name explicitly."
    )
    onnx_file_names = list(Path(model_path).glob("*.onnx"))
    if len(onnx_file_names) == 1:
        return str(onnx_file_names[0])
    elif len(onnx_file_names) > 1:
        raise ValueError(
            f"Multiple .onnx model files found in the model folder {model_path}. Please specify one using the"
            " onnx_file_name argument."
        )
    else:
        raise ValueError(f"No .onnx file found in the model folder {model_path}.")


@model_handler_registry("DistributedOnnxModel")
class DistributedOnnxModelHandler(OliveModelHandler, OnnxEpValidateMixin):
    EXECUTION_PROVIDERS: ClassVar[dict] = {
        "cpu": ["CPUExecutionProvider"],
        "gpu": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }

    DEFAULT_RANKED_MODEL_NAME_FORMAT: ClassVar[str] = "model_{:02d}.onnx"

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
        model_name_pattern: str,
        num_ranks: int,
        inference_settings: Optional[dict] = None,
        use_ort_extensions: bool = False,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.ONNX,
            model_path=model_path,
            model_attributes=model_attributes,
        )
        self.inference_settings = inference_settings
        self.use_ort_extensions = use_ort_extensions

        self.model_name_pattern = model_name_pattern
        self.num_ranks = num_ranks

    def ranked_model_name(self, rank: int) -> str:
        return self.model_name_pattern.format(rank)

    def ranked_model_path(self, rank: int) -> Union[Path, str]:
        return Path(self.model_path) / self.ranked_model_name(rank)

    def load_model(self, rank: int = None) -> ONNXModelHandler:
        return ONNXModelHandler(
            self.ranked_model_path(rank),
            inference_settings=self.inference_settings,
            use_ort_extensions=self.use_ort_extensions,
            model_attributes=self.model_attributes,
        )

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.GPU,  # pylint: disable=signature-differs
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = 0,
    ):
        raise RuntimeError("DistributedOnnxModel doesn't have a session of its own")

    def get_default_execution_providers(self, device: Device):
        """Return a list of supported default execution providers."""
        return ["CPUExecutionProvider"]

    def get_default_execution_providers_with_model(self, filepath: str, device: Device):
        # return firstly available ep as ort default ep
        available_providers = self.get_execution_providers(device)
        for ep in available_providers:
            if self._is_valid_ep(filepath, ep):
                return [ep]

        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def get_execution_providers(device: Device):
        import onnxruntime as ort

        eps_per_device = DistributedOnnxModelHandler.EXECUTION_PROVIDERS.get(device)
        available_providers = ort.get_available_providers()
        return AcceleratorLookup.get_execution_providers(eps_per_device, available_providers)

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update(
            {
                "model_name_pattern": self.model_name_pattern,
                "num_ranks": self.num_ranks,
                "inference_settings": self.inference_settings,
                "use_ort_extensions": self.use_ort_extensions,
            }
        )
        return serialize_to_json(config, check_object=check_object)
