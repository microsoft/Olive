# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import onnx
import onnxruntime as ort
import torch
import transformers
from pydantic import validator

from olive.common.config_utils import ConfigBase, serialize_to_json
from olive.common.user_module_loader import UserModuleLoader
from olive.constants import Framework
from olive.snpe import SNPEDevice, SNPEInferenceSession, SNPESessionOptions
from olive.snpe.tools.dev import get_dlc_metrics
from olive.systems.common import Device

REGISTRY = {}
logger = logging.getLogger(__name__)


class OliveModel(ABC):
    """
    Abstraction for logical "Model", it contains model path and related metadata.
    Each technique accepts Model as input, return Model as output.
    """

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """Register the model."""
        super().__init_subclass__(**kwargs)
        REGISTRY[cls.__name__.lower()] = cls

    def __init__(
        self,
        framework: Framework,
        model_path: Optional[Union[Path, str]] = None,
        name: Optional[str] = None,
        version: Optional[int] = None,
        is_file: bool = False,
        is_aml_model: bool = False,
    ):
        if is_aml_model:
            if not name:
                raise Exception("Please specify model 'name' for Azure ML model")
            if not version:
                raise Exception("Please specify model 'version' for Azure ML model")
            self.model_path = f"azureml:{name}:{version}"
        else:
            self.model_path = model_path
        self.version = version
        self.framework = framework
        self.name = name
        self.is_file = is_file
        self.is_aml_model = is_aml_model

    @abstractmethod
    def load_model(self) -> object:
        """
        Load model from disk, return in-memory model object
        Derived class should implement its specific logic if needed.
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare_session(self, inference_settings: Optional[Dict[str, Any]] = None, device: Device = Device.CPU):
        """
        Prepare inference session for Olive model, return in-memory inference session.
        Derived class should implement its specific logic if needed.
        """
        raise NotImplementedError()

    def to_json(self, check_object: bool = False):
        model_path = self.model_path
        if model_path and Path(model_path).exists():
            model_path = Path(model_path)
        config = {
            "type": self.__class__.__name__,
            "config": {
                "model_path": model_path,
                "name": self.name,
                "is_file": self.is_file,
                "is_aml_model": self.is_aml_model,
                "version": self.version,
            },
        }
        return serialize_to_json(config, check_object)


class ModelConfig(ConfigBase):
    type: str
    config: dict

    @validator("type")
    def validate_type(cls, v):
        if v.lower() not in REGISTRY:
            raise ValueError(f"Unknown model type {v}")
        return v

    def create_model(self):
        return REGISTRY[self.type.lower()](**self.config)


class ONNXModel(OliveModel):

    # device type definition: https://github.com/pytorch/pytorch/blob/master/c10/core/DeviceType.h
    EXECUTION_PROVIDERS = {
        "cpu": ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
        "gpu": [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ],
    }

    def __init__(
        self,
        model_path: str = None,
        name: Optional[str] = None,
        version: Optional[int] = None,
        is_file: bool = True,
        is_aml_model: bool = False,
        inference_settings: Optional[dict] = None,
    ):
        super().__init__(
            framework=Framework.ONNX,
            model_path=model_path,
            name=name,
            version=version,
            is_file=is_file,
            is_aml_model=is_aml_model,
        )
        self.inference_settings = inference_settings

    @staticmethod
    def resolve_path(file_or_dir_path: str, model_filename: str = "model.onnx") -> str:
        """
        The engine provides output paths to ONNX passes that do not contain .onnx
        extension (these paths are generally locations in the cache). This function
        will convert such paths to absolute file paths and also ensure the parent
        directories exist. If the input path is already an ONNX file it is simply
        returned. Examples:

        resolve_path("c:/foo/bar.onnx") -> c:/foo/bar.onnx

        resolve_path("c:/foo/bar") -> c:/foo/bar/model.onnx
        """
        path = Path(file_or_dir_path)
        if path.suffix != ".onnx":
            path = path / model_filename
            parent_dir = path.parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)
        return str(path)

    def load_model(self) -> onnx.ModelProto:
        # HACK: ASSUME no external data
        return onnx.load(self.model_path)

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device):
        sess_options = ort.SessionOptions()
        execution_provider = None
        ort_inference_settings = inference_settings or self.inference_settings
        if ort_inference_settings:
            execution_provider = ort_inference_settings.get("execution_provider")
            session_options = ort_inference_settings.get("session_options")
            inter_op_num_threads = session_options.get("inter_op_num_threads")
            intra_op_num_threads = session_options.get("intra_op_num_threads")
            execution_mode = session_options.get("execution_mode")
            graph_optimization_level = session_options.get("graph_optimization_level")
            extra_session_config = session_options.get("extra_session_config")
            if inter_op_num_threads:
                sess_options.inter_op_num_threads = inter_op_num_threads
            if intra_op_num_threads:
                sess_options.intra_op_num_threads = intra_op_num_threads
            if execution_mode:
                if execution_mode == 0:
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                elif execution_mode == 1:
                    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            if graph_optimization_level:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel(graph_optimization_level)
            if extra_session_config:
                for key, value in extra_session_config.items():
                    sess_options.add_session_config_entry(key, value)

        # if use doesn't not providers ep list, use default value([ep]). Otherwise, use the user's ep list
        if not execution_provider:
            execution_provider = self.get_default_execution_provider(device)
        elif isinstance(execution_provider, list):
            # execution_provider may be a list of tuples where the first item in each tuple is the EP name
            execution_provider = [i[0] if isinstance(i, tuple) else i for i in execution_provider]
        elif isinstance(execution_provider, str):
            execution_provider = [execution_provider]

        if len(execution_provider) >= 1 and execution_provider[0] == "DmlExecutionProvider":
            sess_options.enable_mem_pattern = False

        return ort.InferenceSession(self.model_path, sess_options, providers=execution_provider)

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update({"inference_settings": self.inference_settings})
        return serialize_to_json(config, check_object)

    def is_valid_ep(self, ep: str = None):
        # TODO: should be remove if future accelerators is implemented
        # It should be a bug for onnxruntime where the execution provider is not be fallback.
        try:
            ort.InferenceSession(self.model_path, providers=[ep])
        except Exception as e:
            logger.warning(
                f"Error: {e}Olive will ignore this {ep}."
                + f"Please make sure the environment with {ep} has the required dependencies."
            )
            return False
        return True

    def get_default_execution_provider(self, device: Device):
        # return firstly available ep as ort default ep
        available_providers = self.get_execution_providers(device)
        for ep in available_providers:
            if self.is_valid_ep(ep):
                return [ep]
        return ["CPUExecutionProvider"]

    def get_execution_providers(self, device: Device):
        available_providers = ort.get_available_providers()
        eps_per_device = self.EXECUTION_PROVIDERS.get(device)

        eps = []
        if eps_per_device:
            for ep in available_providers:
                if ep in eps_per_device:
                    eps.append(ep)

        return eps if eps else available_providers


class PyTorchModel(OliveModel):
    def __init__(
        self,
        model_path: str = None,
        name: Optional[str] = None,
        version: Optional[int] = None,
        is_file: bool = False,
        is_aml_model: bool = False,
        model_loader=None,
        model_script=None,
        script_dir=None,
    ):
        if not (
            isinstance(model_loader, Callable)
            or (isinstance(model_loader, str) and model_script)
            or model_path
            or is_aml_model
        ):
            raise ValueError(
                "model_path or is_aml_model is required "
                "since model_loader is not callable or model_script is not provided"
            )

        self.model_loader = model_loader
        self.model_script = model_script
        self.script_dir = script_dir
        self.model = None
        super().__init__(
            framework=Framework.PYTORCH,
            model_path=model_path,
            name=name,
            version=version,
            is_file=is_file,
            is_aml_model=is_aml_model,
        )

    def load_model(self) -> torch.nn.Module:
        if self.model is not None:
            return self.model

        if self.model_loader is not None:
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            model = user_module_loader.call_object(self.model_loader, self.model_path)
        else:
            try:
                model = torch.load(self.model_path)
            except (RuntimeError, ModuleNotFoundError):
                model = torch.jit.load(self.model_path)
        self.model = model

        return model

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device):
        return self.load_model().eval()

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update(
            {
                "model_loader": self.model_loader,
                "model_script": Path(self.model_script) if self.model_script else None,
                "script_dir": Path(self.script_dir) if self.script_dir else None,
            }
        )
        return serialize_to_json(config, check_object)


class SNPEModel(OliveModel):
    def __init__(
        self,
        input_names: List[str],
        input_shapes: List[List[int]],
        output_names: List[str],
        output_shapes: List[List[int]],
        model_path: str = None,
        is_aml_model: bool = False,
        name: Optional[str] = None,
        version: Optional[int] = None,
    ):
        super().__init__(
            framework=Framework.SNPE,
            model_path=model_path,
            name=name,
            version=version,
            is_file=True,
            is_aml_model=is_aml_model,
        )
        self.io_config = {
            "input_names": input_names,
            "input_shapes": input_shapes,
            "output_names": output_names,
            "output_shapes": output_shapes,
        }

    def load_model(self):
        raise NotImplementedError()

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device) -> SNPEInferenceSession:
        session_options = SNPESessionOptions(**inference_settings) if inference_settings else None
        if device == Device.NPU:
            device = SNPEDevice.DSP
        session_options.device = device
        return SNPEInferenceSession(self.model_path, self.io_config, session_options)

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update(self.io_config)
        return serialize_to_json(config, check_object)

    def get_dlc_metrics(self) -> dict:
        return get_dlc_metrics(self.model_path)


class TensorFlowModel(OliveModel):
    def __init__(
        self, model_path: str = None, name: Optional[str] = None, is_file: bool = False, is_aml_model: bool = False
    ):
        super().__init__(
            model_path=model_path, framework=Framework.TENSORFLOW, name=name, is_file=is_file, is_aml_model=is_aml_model
        )

    def load_model(self):
        raise NotImplementedError()

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device):
        raise NotImplementedError()


class OpenVINOModel(OliveModel):
    def __init__(
        self,
        model_path: str,
        name: str = None,
        is_file=False,
        version: Optional[int] = None,
        is_aml_model: bool = False,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.OPENVINO,
            name=name,
            is_file=is_file,
            version=version,
            is_aml_model=is_aml_model,
        )

        if len(list(Path(model_path).glob("*.xml"))) == 0 or len(list(Path(model_path).glob("*.bin"))) == 0:
            raise Exception(f"No OpenVINO model found in {model_path}")
        if len(list(Path(model_path).glob("*.xml"))) > 1 or len(list(Path(model_path).glob("*.bin"))) > 1:
            raise Exception(f"More than 1 OpenVINO models are found in {model_path}")

        for model_file in Path(model_path).glob("*.xml"):
            ov_model = Path(model_file)
        for weights_file in Path(model_path).glob("*.bin"):
            ov_weights = Path(weights_file)

        self.model_config = {
            "model_name": name if name else ov_model.stem,
            "model": str(ov_model.resolve()),
            "weights": str(ov_weights.resolve()),
        }

    def load_model(self):
        try:
            from openvino.tools.pot import load_model
        except ImportError:
            raise ImportError("Please install olive[openvino] to use OpenVINO model")
        return load_model(self.model_config)

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device):
        try:
            from openvino.runtime import Core
        except ImportError:
            raise ImportError("Please install olive[openvino] to use OpenVINO model")
        ie = Core()
        model_pot = ie.read_model(model=self.model_config["model"])
        if device == Device.INTEL_MYRIAD:
            device = "MYRIAD"
        compiled_model = ie.compile_model(model=model_pot, device_name=device.upper())
        return compiled_model


def huggingface_model_loader(model_loader):
    if model_loader is None:
        model_loader = "AutoModel"
    if isinstance(model_loader, str):
        try:
            model_loader = getattr(transformers, model_loader)
        except AttributeError:
            raise AttributeError(f"{model_loader} is not found in transformers")
    elif not isinstance(model_loader, Callable):
        raise ValueError("model_loader must be a callable or a string defined in transformers")

    return model_loader.from_pretrained
