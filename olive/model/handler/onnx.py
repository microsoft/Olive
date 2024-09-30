# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import onnx
from onnx import GraphProto, ModelProto

from olive.common.ort_inference import OrtSessionFallbackError, get_ort_inference_session
from olive.common.utils import load_weights
from olive.constants import Framework, ModelFileFormat
from olive.exception import OliveEvaluationError
from olive.hardware.accelerator import AcceleratorLookup, Device
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.model.handler.mixin import OnnxEpValidateMixin, OnnxGraphMixin
from olive.model.utils.onnx_utils import get_additional_file_path, get_onnx_file_path
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)


@model_handler_registry("ONNXModel")
class ONNXModelHandler(OliveModelHandler, OnnxEpValidateMixin, OnnxGraphMixin):  # pylint: disable=too-many-ancestors
    """ONNX model handler.

    Besides the model loading functionalities, the model handler also provider the onnx graph functionality by mixin

    the mixin class OnnxEpValidateMixin is used to validate the execution providers.
    the mixin class OnnxGraphMixin is used to support onnx graph operations.
    """

    json_config_keys: Tuple[str, ...] = (
        "onnx_file_name",
        "inference_settings",
        "use_ort_extensions",
        "external_initializers_file_name",
        "constant_inputs_file_name",
        "generative",
    )

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        onnx_file_name: Optional[str] = None,
        inference_settings: Optional[dict] = None,
        use_ort_extensions: bool = False,
        model_attributes: Optional[Dict[str, Any]] = None,
        external_initializers_file_name: Optional[str] = None,
        constant_inputs_file_name: Optional[str] = None,
        generative: bool = False,
    ):
        super().__init__(
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.ONNX,
            model_path=model_path,
            model_attributes=model_attributes,
            generative=generative,
        )
        self.inference_settings = inference_settings
        self.use_ort_extensions = use_ort_extensions
        self.onnx_file_name = onnx_file_name
        self.external_initializers_file_name = external_initializers_file_name
        self.constant_inputs_file_name = constant_inputs_file_name

        self.graph = None
        self.all_graphs: Optional[List[GraphProto]] = None

        # check for file names since it will automatically validate the paths
        # these call the property methods which in turn validate the paths using get_onnx_file_path
        # and get_additional_file_path
        to_check = ["model_path", "external_initializers_path", "constant_inputs_path"]
        for attr in to_check:
            getattr(self, attr)

    @property
    def model_path(self) -> str:
        model_path = super().model_path
        return get_onnx_file_path(model_path, self.onnx_file_name) if model_path else None

    @property
    def external_initializers_path(self) -> Optional[str]:
        model_path = super().model_path
        return get_additional_file_path(model_path, self.external_initializers_file_name) if model_path else None

    @property
    def constant_inputs_path(self) -> Optional[str]:
        model_path = super().model_path
        return get_additional_file_path(model_path, self.constant_inputs_file_name) if model_path else None

    def change_model_path_to_dir(self) -> Path:
        """Change the model path to the parent directory of the model file.

        This is used when we want to store more files in the same directory as the model file.
        :return: The parent directory of the model file.
        """
        model_path_resource = Path(self.get_resource("model_path"))
        if model_path_resource.is_dir():
            return model_path_resource

        self.set_resource("model_path", model_path_resource.parent)
        self.onnx_file_name = model_path_resource.name

        return model_path_resource.parent

    def load_model(self, rank: int = None, cache_model: bool = True) -> ModelProto:
        return onnx.load(self.model_path)

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        # user provided inference_settings > model's inference_settings > default settings
        inference_settings = self.merge_inference_settings(inference_settings, execution_providers)
        if not inference_settings["execution_provider"]:
            # if no execution_providers are provided, use the default ones
            inference_settings["execution_provider"] = self._get_default_execution_providers(device)
            inference_settings["provider_options"] = None
        # device id for ranked model
        device_id = rank if device == Device.GPU else None
        # load external initializers if available
        external_initializers = (
            load_weights(self.external_initializers_path) if self.external_initializers_path else None
        )

        try:
            return get_ort_inference_session(
                self.model_path, inference_settings, self.use_ort_extensions, device_id, external_initializers
            )
        except OrtSessionFallbackError as e:
            raise OliveEvaluationError(e) from e

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        output_names = kwargs.pop("output_names", None)
        return session.run(output_names, inputs, **kwargs)

    def merge_inference_settings(
        self, inference_settings: Optional[Dict[str, Any]] = None, execution_providers: List[str] = None
    ):
        """Merge user provided inference settings with model's inference settings.

        user provided inference_settings > model's inference_settings > eps given by arguments
        """
        inference_settings_merged = {"execution_provider": None, "provider_options": None}
        if self.inference_settings:
            # start with model's inference settings
            inference_settings_merged.update(self.inference_settings)
        if inference_settings:
            # update with user provided inference settings
            inference_settings_merged.update(inference_settings)

        if inference_settings_merged.get("execution_provider") is None:
            # use execution providers
            inference_settings_merged["execution_provider"] = execution_providers
            inference_settings_merged["provider_options"] = None

        # execution_provider should be a list
        if isinstance(inference_settings_merged["execution_provider"], (str, tuple)):
            inference_settings_merged["execution_provider"] = [inference_settings_merged["execution_provider"]]
        return inference_settings_merged

    @property
    def io_config(self):
        """Get input/output names, shapes, types of the onnx model without creating an ort session.

        This function loads the onnx model and parses the graph to get the io config.
        """
        if self._io_config:
            return self._io_config

        # save io_config
        self._io_config = self.get_graph_io_config()
        return self._io_config

    def _get_default_execution_providers(self, device: Device):
        # return available ep as ort default ep
        available_providers = AcceleratorLookup.get_execution_providers_for_device(device)
        eps = [ep for ep in available_providers if self.is_valid_ep(self.model_path, ep)]

        if not eps:
            eps.append("CPUExecutionProvider")
        return eps


@model_handler_registry("DistributedOnnxModel")
class DistributedOnnxModelHandler(OliveModelHandler, OnnxEpValidateMixin):
    json_config_keys: Tuple[str, ...] = (
        "model_name_pattern",
        "num_ranks",
        "inference_settings",
        "use_ort_extensions",
    )

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
        generative: bool = False,
    ):
        super().__init__(
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.ONNX,
            model_path=model_path,
            model_attributes=model_attributes,
            generative=generative,
        )
        self.inference_settings = inference_settings
        self.use_ort_extensions = use_ort_extensions

        self.model_name_pattern = model_name_pattern
        self.num_ranks = num_ranks

    def ranked_model_name(self, rank: int) -> str:
        return self.model_name_pattern.format(rank)

    def ranked_model_path(self, rank: int) -> Union[Path, str]:
        return Path(self.model_path) / self.ranked_model_name(rank)

    def load_model(self, rank: int = None, cache_model: bool = True) -> ONNXModelHandler:
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

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        raise RuntimeError("DistributedOnnxModel doesn't have a session of its own")

    def get_default_execution_providers_with_model(self, filepath: str, device: Device):
        # return firstly available ep as ort default ep
        available_providers = self.get_execution_providers(device)
        for ep in available_providers:
            if self.is_valid_ep(filepath, ep):
                return [ep]

        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def get_execution_providers(device: Device):
        import onnxruntime as ort

        eps_per_device = DistributedOnnxModelHandler.EXECUTION_PROVIDERS.get(device)
        available_providers = ort.get_available_providers()
        return AcceleratorLookup.get_execution_providers(eps_per_device, available_providers)
