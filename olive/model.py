# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import onnx
import torch
from pydantic import validator

from olive.common.config_utils import ConfigBase, serialize_to_json, validate_config
from olive.common.ort_inference import get_ort_inference_session
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
    def load_model(self, rank: int = None) -> object:
        """
        Load model from disk, return in-memory model object
        Derived class should implement its specific logic if needed.
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare_session(
        self, inference_settings: Optional[Dict[str, Any]] = None, device: Device = Device.CPU, rank: int = None
    ):
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


class IOConfig(ConfigBase):
    input_names: List[str]
    input_shapes: List[List[int]] = None
    input_types: List[str] = None
    output_names: List[str]
    output_shapes: List[List[int]] = None
    output_types: List[str] = None

    @validator("input_shapes", "input_types")
    def check_input_shapes(cls, v, values):
        if "input_names" not in values:
            raise ValueError("Invalid input_names")
        if len(v) != len(values["input_names"]):
            raise ValueError("input_names and input_shapes must have the same length")
        return v

    @validator("output_shapes", "output_types")
    def check_output_shapes(cls, v, values):
        if "output_names" not in values:
            raise ValueError("Invalid output_names")
        if len(v) != len(values["output_names"]):
            raise ValueError("output_names and output_shapes must have the same length")
        return v


class ONNXModelBase(OliveModel):
    """
    Abstract class to manage ONNX models
    """

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
    def _is_valid_ep(filepath: str, ep: str = None):
        # TODO: should be remove if future accelerators is implemented
        # It should be a bug for onnxruntime where the execution provider is not be fallback.
        import onnxruntime as ort

        try:
            ort.InferenceSession(filepath, providers=[ep])
        except Exception as e:
            logger.warning(
                f"Error: {e}Olive will ignore this {ep}."
                + f"Please make sure the environment with {ep} has the required dependencies."
            )
            return False
        return True

    @abstractmethod
    def get_default_execution_providers(self, device: Device):
        """
        Returns a list of supported default execution providers
        """
        return ["CPUExecutionProvider"]


class ONNXModel(ONNXModelBase):
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
        "npu": ["QNNExecutionProvider", "CPUExecutionProvider"],
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
            model_path=model_path,
            name=name,
            version=version,
            is_file=is_file,
            is_aml_model=is_aml_model,
            inference_settings=inference_settings,
        )
        self.io_config = None

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

    def load_model(self, rank: int = None) -> onnx.ModelProto:
        # HACK: ASSUME no external data
        return onnx.load(self.model_path)

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device, rank: int = None):
        # user provided inference_settings > model's inference_settings > default settings
        inference_settings = inference_settings or self.inference_settings or {}
        # deep copy to avoid modifying the original settings
        inference_settings = deepcopy(inference_settings)

        # if user doesn't not provide ep list, use default value([ep]). Otherwise, use the user's ep list
        if not inference_settings.get("execution_provider"):
            inference_settings["execution_provider"] = self.get_default_execution_providers(device)

        return get_ort_inference_session(self.model_path, inference_settings)

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update({"inference_settings": self.inference_settings})
        return serialize_to_json(config, check_object)

    def get_default_execution_providers(self, device: Device):
        # return firstly available ep as ort default ep
        available_providers = ONNXModel.get_execution_providers(device)
        for ep in available_providers:
            if ONNXModelBase._is_valid_ep(self.model_path, ep):
                return [ep]
        return super().get_default_execution_providers(device)

    @staticmethod
    def get_execution_providers(device: Device):
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        eps_per_device = ONNXModel.EXECUTION_PROVIDERS.get(device)

        eps = []
        if eps_per_device:
            for ep in available_providers:
                if ep in eps_per_device:
                    eps.append(ep)

        return eps if eps else available_providers

    def get_io_config(self):
        """
        Get input/output names, shapes, types of the onnx model without creating an ort session.
        This function loads the onnx model and parses the graph to get the io config.
        """
        if self.io_config:
            return self.io_config

        try:
            from onnx.helper import tensor_dtype_to_np_dtype
        except ImportError:
            from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

            def tensor_dtype_to_np_dtype(tensor_type):
                return TENSOR_TYPE_TO_NP_TYPE[tensor_type]

        # external data is not needed for io config parsing
        # the .onnx model already contains all of the graph information
        # this method works whether the external data is in the same directory or not
        model = onnx.load(self.model_path, load_external_data=False)
        io_config = {
            "input_names": [],
            "input_shapes": [],
            "input_types": [],
            "output_names": [],
            "output_shapes": [],
            "output_types": [],
        }
        for prefix, ios in [("input", model.graph.input), ("output", model.graph.output)]:
            for io in ios:
                # get name, type, shape
                name = io.name
                tensor_type = io.type.tensor_type
                data_type = str(tensor_dtype_to_np_dtype(tensor_type.elem_type))
                shape = [dim.dim_param if dim.dim_param else dim.dim_value for dim in tensor_type.shape.dim]

                # append to io_config
                io_config[f"{prefix}_names"].append(name)
                io_config[f"{prefix}_types"].append(data_type)
                io_config[f"{prefix}_shapes"].append(shape)

        # save io_config
        self.io_config = io_config

        return io_config


class PyTorchModel(OliveModel):
    def __init__(
        self,
        model_path: str = None,
        name: Optional[str] = None,
        version: Optional[int] = None,
        is_file: bool = False,
        is_aml_model: bool = False,
        model_loader: Union[str, Callable] = None,
        model_script: Union[str, Path] = None,
        script_dir: Union[str, Path] = None,
        io_config: Union[Dict[str, Any], IOConfig] = None,
        dummy_input_func: Union[str, Callable] = None,
        dynamic_axes: Dict[str, Dict[int, str]] = None,
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

        # io config for conversion to onnx
        self.io_config = validate_config(io_config, IOConfig)
        self.dummy_inputs_func = dummy_input_func

        # dynamic axes for conversion to onnx
        self.dynamic_axes = dynamic_axes
        # during json serialization, the int keys in dynamic_axes will be converted to str
        # so we need to convert them back to int when deserializing
        if self.dynamic_axes:
            for k, v in self.dynamic_axes.items():
                self.dynamic_axes[k] = {int(kk): vv for kk, vv in v.items()}

        self.dummy_inputs = None

    def load_model(self, rank: int = None) -> torch.nn.Module:
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

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device, rank: int = None):
        return self.load_model().eval()

    def get_dummy_inputs(self):
        """
        Return a dummy input for the model.
        """
        if self.dummy_inputs is not None:
            return self.dummy_inputs

        assert self.dummy_inputs_func or (
            self.io_config and self.io_config.input_shapes
        ), "dummy_input_func or io_config.input_shapes must be provided to get dummy input"

        if self.dummy_inputs_func is not None:
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            self.dummy_inputs = user_module_loader.call_object(self.dummy_inputs_func, self)
        else:
            str_to_type = {
                "float32": torch.float32,
                "float16": torch.float16,
                "int32": torch.int32,
                "int64": torch.int64,
                "int8": torch.int8,
                "bool": torch.bool,
            }
            input_types = self.io_config.input_types or ["float32"] * len(self.io_config.input_shapes)
            dummy_inputs = []
            for shape, dtype in zip(self.io_config.input_shapes, input_types):
                dummy_inputs.append(torch.zeros(shape, dtype=str_to_type[dtype]))
            dummy_inputs = tuple(dummy_inputs) if len(dummy_inputs) > 1 else dummy_inputs[0]

        self.dummy_inputs = dummy_inputs

        return dummy_inputs

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update(
            {
                "model_loader": self.model_loader,
                "model_script": Path(self.model_script) if self.model_script else None,
                "script_dir": Path(self.script_dir) if self.script_dir else None,
                "io_config": self.io_config,
                "dummy_input_func": self.dummy_inputs_func,
                "dynamic_axes": self.dynamic_axes,
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

    def load_model(self, rank: int = None):
        raise NotImplementedError()

    def prepare_session(
        self, inference_settings: Dict[str, Any], device: Device, rank: int = None
    ) -> SNPEInferenceSession:
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

    def load_model(self, rank: int = None):
        raise NotImplementedError()

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device, rank: int = None):
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

    def load_model(self, rank: int = None):
        try:
            from openvino.tools.pot import load_model
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model")
        return load_model(self.model_config)

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device, rank: int = None):
        try:
            from openvino.runtime import Core
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model")
        ie = Core()
        model_pot = ie.read_model(model=self.model_config["model"])
        if device == Device.INTEL_MYRIAD:
            device = "MYRIAD"
        compiled_model = ie.compile_model(model=model_pot, device_name=device.upper())
        return compiled_model


def huggingface_model_loader(model_loader):
    import transformers

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


class DistributedOnnxModel(ONNXModelBase):
    EXECUTION_PROVIDERS = {
        "cpu": ["CPUExecutionProvider"],
        "gpu": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }

    def __init__(
        self,
        model_filepaths: List[str],
        name: Optional[str] = None,
        version: Optional[int] = None,
        inference_settings: Optional[dict] = None,
    ):
        super().__init__(
            model_path=None,
            name=name,
            version=version,
            is_file=False,
            is_aml_model=False,
            inference_settings=inference_settings,
        )
        self.model_filepaths = model_filepaths

    @property
    def ranks(self):
        return len(self.model_filepaths)

    def load_model(self, rank: int) -> ONNXModel:
        return ONNXModel(self.model_filepaths[rank], inference_settings=self.inference_settings)

    def prepare_session(
        self, inference_settings: Optional[Dict[str, Any]] = None, device: Device = Device.GPU, rank: int = 0
    ):
        # user provided inference_settings > model's inference_settings > default settings
        inference_settings = inference_settings or self.inference_settings or {}
        # deep copy to avoid modifying the original settings
        inference_settings = deepcopy(inference_settings)

        # if user doesn't not provide ep list, use default value([ep]). Otherwise, use the user's ep list
        execution_providers = inference_settings.get("execution_provider")
        if not execution_providers:
            execution_providers = self.get_default_execution_providers(device)
            inference_settings["execution_provider"] = execution_providers

        if not inference_settings.get("provider_options"):
            inference_settings["provider_options"] = [
                {"device_id": str(rank)} if ep == "CUDAExecutionProvider" else {} for ep in execution_providers
            ]

        return get_ort_inference_session(self.model_filepaths[rank], inference_settings)

    def get_default_execution_providers(self, filepath: str, device: Device):
        # return firstly available ep as ort default ep
        available_providers = DistributedOnnxModel.get_execution_providers(device)
        for ep in available_providers:
            if ONNXModelBase._is_valid_ep(filepath, ep):
                return [ep]

        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def get_execution_providers(device: Device):
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        eps_per_device = DistributedOnnxModel.EXECUTION_PROVIDERS.get(device)

        eps = []
        if eps_per_device:
            for ep in available_providers:
                if ep in eps_per_device:
                    eps.append(ep)

        return eps if eps else available_providers

    def to_json(self, check_object: bool = False):
        config = {
            "type": self.__class__.__name__,
            "config": {
                "model_filepaths": self.model_filepaths,
                "name": self.name,
                "version": self.version,
                "inference_settings": self.inference_settings,
            },
        }
        return serialize_to_json(config, check_object=check_object)
