# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import onnx
import torch
import yaml
from onnx import AttributeProto, GraphProto
from pydantic import validator

from olive.common.config_utils import ConfigBase, serialize_to_json, validate_config
from olive.common.ort_inference import get_ort_inference_session
from olive.common.user_module_loader import UserModuleLoader
from olive.constants import Framework, ModelFileFormat
from olive.hf_utils import (
    huggingface_model_loader,
    load_huggingface_model_from_model_class,
    load_huggingface_model_from_task,
)
from olive.snpe import SNPEDevice, SNPEInferenceSession, SNPESessionOptions
from olive.snpe.tools.dev import get_dlc_metrics
from olive.systems.common import Device

REGISTRY = {}
logger = logging.getLogger(__name__)


class ModelStorageKind(str, Enum):
    LocalFile = "file"
    LocalFolder = "folder"
    AzureMLModel = "azureml"

    def __str__(self) -> str:
        return self.value


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
        model_file_format: ModelFileFormat,
        model_path: Optional[Union[Path, str]] = None,
        name: Optional[str] = None,
        version: Optional[int] = None,
        model_storage_kind: ModelStorageKind = ModelStorageKind.LocalFile,
    ):
        if isinstance(model_storage_kind, str):
            model_storage_kind = ModelStorageKind(model_storage_kind)

        assert isinstance(model_storage_kind, ModelStorageKind)

        if model_storage_kind == ModelStorageKind.AzureMLModel:
            if not name:
                raise Exception("Please specify model 'name' for Azure ML model")
            if not version:
                raise Exception("Please specify model 'version' for Azure ML model")
            self.model_path = f"azureml:{name}:{version}"
        else:
            self.model_path = model_path
        self.version = version
        self.framework = framework
        self.model_file_format = model_file_format
        self.name = name
        self.composite_parent = None
        self.model_storage_kind = model_storage_kind

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

    def set_composite_parent(self, cp):
        self.composite_parent = cp

    def get_composite_parent(self):
        return self.composite_parent

    def to_json(self, check_object: bool = False):
        model_path = self.model_path
        if model_path and Path(model_path).exists():
            model_path = Path(model_path)
        config = {
            "type": self.__class__.__name__,
            "config": {
                "model_path": model_path,
                "name": self.name,
                "model_storage_kind": self.model_storage_kind.value,
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
    dynamic_axes: Dict[str, Dict[int, str]] = None

    @validator("input_shapes", "input_types")
    def check_input_shapes(cls, v, values):
        if not v:
            return v

        if "input_names" not in values:
            raise ValueError("Invalid input_names")
        if len(v) != len(values["input_names"]):
            raise ValueError("input_names and input_shapes must have the same length")
        return v

    @validator("output_shapes", "output_types")
    def check_output_shapes(cls, v, values):
        if not v:
            return v

        if "output_names" not in values:
            raise ValueError("Invalid output_names")
        if len(v) != len(values["output_names"]):
            raise ValueError("output_names and output_shapes must have the same length")
        return v

    @validator("dynamic_axes")
    def convert_dynamic_axes(cls, v):
        if not v:
            return v

        dynamic_axes = v
        for k, v in dynamic_axes.items():
            dynamic_axes[k] = {int(kk): vv for kk, vv in v.items()}
        return dynamic_axes


class HFComponent(ConfigBase):
    name: str
    io_config: IOConfig = None
    dummy_inputs_func: Union[str, Callable] = None


class HFConfig(ConfigBase):
    model_name: str
    task: str = None
    # TODO: remove model_class and only use task
    model_class: str = None
    use_ort_implementation: bool = False
    components: List[HFComponent] = None

    @validator("model_class", always=True)
    def task_or_model_class_required(cls, v, values):
        if "task" not in values:
            raise ValueError("Invalid task")

        if not v and not values["task"]:
            raise ValueError("Either task or model_class must be specified")
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
        model_storage_kind: Union[str, ModelStorageKind] = ModelStorageKind.LocalFile,
        inference_settings: Optional[dict] = None,
    ):
        super().__init__(
            framework=Framework.ONNX,
            model_file_format=ModelFileFormat.ONNX,
            model_path=model_path,
            name=name,
            version=version,
            model_storage_kind=model_storage_kind,
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
        model_storage_kind: Union[str, ModelStorageKind] = ModelStorageKind.LocalFile,
        inference_settings: Optional[dict] = None,
    ):

        super().__init__(
            model_path=model_path,
            name=name,
            version=version,
            model_storage_kind=model_storage_kind,
            inference_settings=inference_settings,
        )
        self.io_config = None
        self.graph = None
        self.all_graphs: Optional[List[GraphProto]] = None

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

    def nodes(self):
        for graph in self.get_all_graphs():
            for node in graph.node:
                yield node

    def get_graph(self):
        if self.graph is not None:
            return self.graph
        self.graph = self.load_model().graph
        return self.graph

    def get_all_graphs(self):
        if self.all_graphs is not None:
            return self.all_graphs
        self.all_graphs = []
        graph_queue = [self.get_graph()]
        while graph_queue:
            graph = graph_queue.pop(0)
            self.all_graphs.append(graph)
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == AttributeProto.AttributeType.GRAPH:
                        assert isinstance(attr.g, GraphProto)
                        graph_queue.append(attr.g)
                    if attr.type == AttributeProto.AttributeType.GRAPHS:
                        for g in attr.graphs:
                            assert isinstance(g, GraphProto)
                            graph_queue.append(g)
        return self.all_graphs

    def output_name_to_node(self):
        output_name_to_node = {}
        for node in self.nodes():
            for output_name in node.output:
                if output_name:  # could be empty when it is optional
                    output_name_to_node[output_name] = node
        return output_name_to_node

    def get_initializer(self, name):
        for graph in self.get_all_graphs():
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor
        return None

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
                if tensor_type.elem_type == 0:
                    # sequence type
                    # TODO: add support for different types
                    # refer to https://github.com/lutzroeder/netron/blob/main/source/onnx.js#L1424
                    tensor_type = io.type.sequence_type.elem_type.tensor_type
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
        model_file_format: ModelFileFormat = ModelFileFormat.PYTORCH_ENTIRE_MODEL,
        name: Optional[str] = None,
        version: Optional[int] = None,
        model_storage_kind: Union[str, ModelStorageKind] = ModelStorageKind.LocalFolder,
        model_loader: Union[str, Callable] = None,
        model_script: Union[str, Path] = None,
        script_dir: Union[str, Path] = None,
        io_config: Union[Dict[str, Any], IOConfig] = None,
        dummy_inputs_func: Union[str, Callable] = None,
        hf_config: Union[Dict[str, Any], HFConfig] = None,
    ):
        if not (
            isinstance(model_loader, Callable)
            or (isinstance(model_loader, str) and model_script)
            or model_path
            or model_storage_kind == ModelStorageKind.AzureMLModel
            or hf_config
        ):
            raise ValueError(
                "model_path or model_storage_kind/AzureMLModel is required "
                "since model_loader is not callable or model_script is not provided"
            )

        self.model_loader = model_loader
        self.model_script = model_script
        self.script_dir = script_dir
        self.model = None
        super().__init__(
            framework=Framework.PYTORCH,
            model_file_format=model_file_format,
            model_path=model_path,
            name=name,
            version=version,
            model_storage_kind=model_storage_kind,
        )

        # io config for conversion to onnx
        self.io_config = validate_config(io_config, IOConfig) if io_config else None
        self.dummy_inputs_func = dummy_inputs_func

        self.dummy_inputs = None

        # huggingface config
        self.hf_config = validate_config(hf_config, HFConfig) if hf_config else None

    def load_model(self, rank: int = None) -> torch.nn.Module:
        if self.model is not None:
            return self.model

        if self.model_loader is not None:
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            model = user_module_loader.call_object(self.model_loader, self.model_path)
        elif self.hf_config is not None:
            if self.hf_config.task:
                model = load_huggingface_model_from_task(self.hf_config.task, self.hf_config.model_name)
            else:
                model = load_huggingface_model_from_model_class(
                    self.hf_config.model_class, self.hf_config.model_name, self.hf_config.use_ort_implementation
                )
        else:
            if self.model_file_format == ModelFileFormat.PYTORCH_ENTIRE_MODEL:
                model = torch.load(self.model_path)
            elif self.model_file_format == ModelFileFormat.PYTORCH_TORCH_SCRIPT:
                model = torch.jit.load(self.model_path)
            elif self.model_file_format == ModelFileFormat.PYTORCH_MLFLOW_MODEL:
                model = self.load_mlflow_model()
            elif self.model_file_format == ModelFileFormat.PYTORCH_STATE_DICT:
                raise ValueError("Please use customized model loader to load state dict model.")
            else:
                raise ValueError(f"Unsupported model file format: {self.model_file_format}")

        self.model = model

        return model

    def load_mlflow_model(self):
        tmp_dir = tempfile.TemporaryDirectory(prefix="mlflow_tmp")
        tmp_dir_path = Path(tmp_dir.name)

        shutil.copytree(os.path.join(self.model_path, "data/model"), tmp_dir_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(self.model_path, "data/config"), tmp_dir_path, dirs_exist_ok=True)
        shutil.copytree(os.path.join(self.model_path, "data/tokenizer"), tmp_dir_path, dirs_exist_ok=True)

        with open(os.path.join(self.model_path, "MLmodel"), "r") as fp:
            mlflow_data = yaml.safe_load(fp)
            hf_pretrained_class = mlflow_data["flavors"]["hftransformers"]["hf_pretrained_class"]

        model_loader = huggingface_model_loader(hf_pretrained_class)
        loaded_model = model_loader(tmp_dir_path)
        loaded_model.eval()

        tmp_dir.cleanup()

        return loaded_model

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device, rank: int = None):
        return self.load_model().eval()

    # TODO: remove this method once we have olive datasets implemented.
    # The dataset should be able to provide the dummy inputs.
    def get_dummy_inputs(self):
        """
        Return a dummy input for the model.
        """
        if self.dummy_inputs is not None:
            return self.dummy_inputs

        assert self.dummy_inputs_func or (
            self.io_config and self.io_config.input_shapes
        ), "dummy_inputs_func or io_config.input_shapes must be provided to get dummy input"

        if self.dummy_inputs_func is not None:
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            dummy_inputs = user_module_loader.call_object(self.dummy_inputs_func, self)
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

    @property
    def components(self) -> List[str]:
        """
        Names of the components of the model.
        """
        if not self.hf_config or not self.hf_config.components:
            return None

        return [component.name for component in self.hf_config.components]

    def get_component(self, component_name: str) -> "PyTorchModel":
        """
        Get a component of the model as a PyTorchModel.
        """
        assert self.components, "hf_config.components must be provided to get component"
        assert component_name in self.components, f"component {component_name} not found in hf_config"

        model = self.load_model()
        model_component = getattr(model, component_name)

        # get the component from hf_config
        components_dict = {component.name: component for component in self.hf_config.components}
        hf_component = components_dict[component_name]

        def model_loader(_):
            return model_component

        return PyTorchModel(
            model_loader=model_loader,
            name=hf_component.name,
            io_config=hf_component.io_config,
            dummy_inputs_func=hf_component.dummy_inputs_func,
            model_script=self.model_script,
            script_dir=self.script_dir,
        )

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update(
            {
                "model_loader": self.model_loader,
                "model_script": Path(self.model_script) if self.model_script else None,
                "script_dir": Path(self.script_dir) if self.script_dir else None,
                "io_config": self.io_config,
                "dummy_inputs_func": self.dummy_inputs_func,
                "hf_config": self.hf_config,
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
        model_storage_kind=ModelStorageKind.LocalFile,
        name: Optional[str] = None,
        version: Optional[int] = None,
    ):
        super().__init__(
            framework=Framework.SNPE,
            model_file_format=ModelFileFormat.SNPE_DLC,
            model_path=model_path,
            name=name,
            version=version,
            model_storage_kind=model_storage_kind,
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
        self,
        model_path: str = None,
        model_file_format: ModelFileFormat = ModelFileFormat.TENSORFLOW_SAVED_MODEL,
        name: Optional[str] = None,
        model_storage_kind=ModelStorageKind.LocalFolder,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.TENSORFLOW,
            model_file_format=model_file_format,
            name=name,
            model_storage_kind=model_storage_kind,
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
        model_storage_kind=ModelStorageKind.LocalFolder,
        version: Optional[int] = None,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.OPENVINO,
            model_file_format=ModelFileFormat.OPENVINO_IR,
            name=name,
            version=version,
            model_storage_kind=model_storage_kind,
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
            model_storage_kind=ModelStorageKind.LocalFolder,
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


class CompositeOnnxModel(ONNXModelBase):
    """
    CompositeOnnxModel represents multi component models. Whisper is an example composite
    model that has encoder and decoder components. CompositeOnnxModel is a collection of
    OnnxModels.
    """

    def __init__(
        self,
        model_components: List[str],
        name: Optional[str] = None,
        version: Optional[int] = None,
    ):
        super().__init__(model_path=None, name=name, version=version, model_storage_kind=ModelStorageKind.LocalFolder)

        if isinstance(model_components[0], dict):
            assert all(
                [m.get("type").lower() == "onnxmodel" for m in model_components]
            ), "All components must be ONNXModel"
            self.model_components = [ONNXModel(**m.get("config", {})) for m in model_components]
        else:
            assert all([isinstance(m, ONNXModel) for m in model_components]), "All components must be ONNXModel"
            self.model_components = model_components

        for m in self.model_components:
            m.set_composite_parent(self)

    def load_model(self, rank: int = None):
        raise NotImplementedError()

    def prepare_session(self, inference_settings: Dict[str, Any], device: Device, rank: int = None):
        raise NotImplementedError()

    def get_default_execution_providers(self, device: Device):
        raise NotImplementedError()

    def get_model_components(self):
        return self.model_components

    def to_json(self, check_object: bool = False):
        json_dict = {
            "type": self.__class__.__name__,
            "config": {
                "name": self.name,
                "version": self.version,
            },
        }
        json_dict["config"]["model_components"] = []
        for m in self.model_components:
            json_dict["config"]["model_components"].append(m.to_json(check_object))

        return serialize_to_json(json_dict, check_object)
