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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import onnx
import torch
import yaml
from onnx import AttributeProto, GraphProto
from pydantic import validator

import olive.data.template as data_config_template
from olive.common.config_utils import ConfigBase, serialize_to_json, validate_config
from olive.common.ort_inference import get_ort_inference_session
from olive.common.user_module_loader import UserModuleLoader
from olive.constants import Framework, ModelFileFormat
from olive.hardware import AcceleratorLookup, Device
from olive.model.hf_utils import HFConfig, get_hf_model_dummy_input, huggingface_model_loader
from olive.model.model_config import IOConfig
from olive.resource_path import (
    OLIVE_RESOURCE_ANNOTATIONS,
    ResourcePath,
    ResourcePathConfig,
    ResourceType,
    create_resource_path,
)
from olive.snpe import SNPEDevice, SNPEInferenceSession, SNPESessionOptions
from olive.snpe.tools.dev import get_dlc_metrics

REGISTRY = {}
logger = logging.getLogger(__name__)


class OliveModel(ABC):
    """
    Abstraction for logical "Model", it contains model path and related metadata.
    Each technique accepts Model as input, return Model as output.
    """

    resource_keys = ["model_path"]

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """Register the model."""
        super().__init_subclass__(**kwargs)
        REGISTRY[cls.__name__.lower()] = cls

    def __init__(
        self,
        framework: Framework,
        model_file_format: ModelFileFormat,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        self.framework = framework
        self.model_file_format = model_file_format
        self.composite_parent = None
        self.model_attributes = model_attributes
        self.io_config = None
        # store resource paths
        self.resource_paths: Dict[str, str] = {}
        resources = {}
        resources["model_path"] = model_path
        self.add_resources(resources)

    @property
    def model_path(self) -> str:
        """Return local model path."""
        return self.get_resource("model_path")

    def add_resources(self, resources: Dict[str, OLIVE_RESOURCE_ANNOTATIONS]):
        for resource_name, resource_path in resources.items():
            if resource_path is not None:
                resolved_resource_path = create_resource_path(resource_path)
                assert (
                    resolved_resource_path.is_local_resource_or_string_name()
                ), f"{resource_name} must be local path or string name."
                self.resource_paths[resource_name] = resolved_resource_path.get_path()
            else:
                self.resource_paths[resource_name] = None

    def set_resource(self, resource_name: str, resource_path: Union[Path, str, ResourcePath, ResourcePathConfig]):
        """
        Set resource path.

        :param resource_name: name of the resource.
        :param resource_path: resource path.
        """
        if resource_name not in self.resource_paths:
            raise ValueError(f"{resource_name} is not a valid resource name.")
        if self.resource_paths[resource_name]:
            logger.debug(f"Overriding {resource_name} from {self.resource_paths[resource_name]} to {resource_path}.")

        if resource_path is not None:
            resolved_resource_path = create_resource_path(resource_path)
            assert (
                resolved_resource_path.is_local_resource_or_string_name()
            ), f"{resource_name} must be local path or string name."
            resource_path = resolved_resource_path.get_path()

        self.resource_paths[resource_name] = resource_path

    def get_resource(self, resource_name: str) -> str:
        """
        Get local path of a resource.

        :param resource_name: name of the resource.
        :return: local path.
        """
        assert resource_name in self.resource_paths, f"{resource_name} is not a valid resource name."
        resource = self.resource_paths[resource_name]
        assert resource is None or isinstance(resource, str)
        return resource

    @abstractmethod
    def load_model(self, rank: int = None) -> object:
        """
        Load model from disk, return in-memory model object
        Derived class should implement its specific logic if needed.
        """
        raise NotImplementedError()

    @abstractmethod
    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
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

    def get_io_config(self) -> Dict[str, Any]:
        return self.io_config

    def to_json(self, check_object: bool = False):
        config = {
            "type": self.__class__.__name__,
            "config": {
                # serialize resource paths
                resource_name: resource_path if resource_path else None
                for resource_name, resource_path in self.resource_paths.items()
            },
        }
        config["config"].update({"model_attributes": self.model_attributes})
        return serialize_to_json(config, check_object)


class ModelConfig(ConfigBase):
    type: str
    config: dict

    @validator("type")
    def validate_type(cls, v):
        if v.lower() not in REGISTRY:
            raise ValueError(f"Unknown model type {v}")
        return v

    def get_resource_keys(self):
        cls = REGISTRY[self.type.lower()]
        return cls.resource_keys

    def get_resource_paths(self):
        resource_keys = self.get_resource_keys()
        return {k: create_resource_path(v) for k, v in self.config.items() if k in resource_keys}

    def create_model(self):
        return REGISTRY[self.type.lower()](**self.config)


class ONNXModelBase(OliveModel):
    """
    Abstract class to manage ONNX models
    """

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
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

    def _is_valid_ep(self, filepath: str, ep: str = None):
        # TODO: should be remove if future accelerators is implemented
        # It should be a bug for onnxruntime where the execution provider is not be fallback.
        import onnxruntime as ort

        try:
            sess_options = ort.SessionOptions()
            if self.use_ort_extensions:
                # register custom ops for onnxruntime-extensions
                from onnxruntime_extensions import get_library_path

                sess_options.register_custom_ops_library(get_library_path())

            ort.InferenceSession(filepath, sess_options, providers=[ep])
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
    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        onnx_file_name: Optional[str] = None,
        inference_settings: Optional[dict] = None,
        use_ort_extensions: bool = False,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=model_path,
            inference_settings=inference_settings,
            use_ort_extensions=use_ort_extensions,
            model_attributes=model_attributes,
        )
        self.onnx_file_name = onnx_file_name
        self.io_config = None
        self.graph = None
        self.all_graphs: Optional[List[GraphProto]] = None

        # check for onnx file name since it will do validation
        _ = self.model_path

    @staticmethod
    def get_onnx_file_path(model_path: str, onnx_file_name: Optional[str] = None) -> str:
        """
        Get the path to the ONNX model file. If model_path is a file, it is returned as is. If model_path is a
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

    @property
    def model_path(self) -> str:
        model_path = super().model_path
        model_path = self.get_onnx_file_path(model_path, self.onnx_file_name) if model_path else None
        return model_path

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
        return onnx.load(self.model_path)

    def prepare_session(
        self,
        inference_settings: Dict[str, Any],
        device: Device,
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
        config["config"].update(
            {
                "onnx_file_name": self.onnx_file_name,
                "inference_settings": self.inference_settings,
                "use_ort_extensions": self.use_ort_extensions,
            }
        )
        return serialize_to_json(config, check_object)

    def get_default_execution_providers(self, device: Device):
        # return firstly available ep as ort default ep
        available_providers = AcceleratorLookup.get_execution_providers_for_device(device)
        for ep in available_providers:
            if self._is_valid_ep(self.model_path, ep):
                return [ep]
        return super().get_default_execution_providers(device)

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

        return self.io_config


class PyTorchModel(OliveModel):
    resource_keys = ["model_path", "script_dir", "model_script", "adapter_path"]

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_file_format: ModelFileFormat = ModelFileFormat.PYTORCH_ENTIRE_MODEL,
        model_loader: Union[str, Callable] = None,
        model_script: Union[str, Path] = None,
        script_dir: Union[str, Path] = None,
        io_config: Union[Dict[str, Any], IOConfig] = None,
        dummy_inputs_func: Union[str, Callable] = None,
        hf_config: Union[Dict[str, Any], HFConfig] = None,
        adapter_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        if not (
            isinstance(model_loader, Callable)
            or (isinstance(model_loader, str) and model_script)
            or model_path
            or hf_config
        ):
            raise ValueError(
                "model_path is required since model_loader is not callable or model_script is not provided"
            )

        self.model_loader = model_loader
        self.model = None
        super().__init__(
            framework=Framework.PYTORCH,
            model_file_format=model_file_format,
            model_path=model_path,
            model_attributes=model_attributes,
        )
        resources = {"adapter_path": adapter_path, "script_dir": script_dir, "model_script": model_script}
        self.add_resources(resources)

        self.hf_config = None
        if hf_config:
            self.hf_config = validate_config(hf_config, HFConfig)
            hf_model_config = self.get_hf_model_config().to_dict()
            model_attr = self.model_attributes or {}
            hf_model_config.update(model_attr)
            self.model_attributes = hf_model_config

        # ensure that script_dirs are local folder
        script_dir_resource = create_resource_path(self.script_dir)
        if script_dir_resource:
            assert script_dir_resource.type == ResourceType.LocalFolder, "script_dir must be a local directory."

        # ensure that model_script is local file or string name
        model_script_resource = create_resource_path(self.model_script)
        if model_script_resource:
            assert model_script_resource.type in (
                ResourceType.LocalFile,
                ResourceType.StringName,
            ), "model_script must be a local file or a string name."

        # io config for conversion to onnx
        self.io_config = validate_config(io_config, IOConfig).dict() if io_config else None
        self.dummy_inputs_func = dummy_inputs_func

        self.dummy_inputs = None

    @property
    def script_dir(self) -> str:
        return self.get_resource("script_dir")

    @property
    def model_script(self) -> str:
        return self.get_resource("model_script")

    def load_model(self, rank: int = None) -> torch.nn.Module:
        if self.model is not None:
            return self.model

        if self.model_loader is not None:
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            model = user_module_loader.call_object(self.model_loader, self.model_path)
        elif self.hf_config and (self.hf_config.model_class or self.hf_config.task):
            model = self.hf_config.load_model(self.model_path)
        else:
            if self.model_file_format == ModelFileFormat.PYTORCH_ENTIRE_MODEL:
                model = torch.load(self.model_path)
            elif self.model_file_format == ModelFileFormat.PYTORCH_TORCH_SCRIPT:
                model = torch.jit.load(self.model_path)
            elif self.model_file_format == ModelFileFormat.PYTORCH_MLFLOW_MODEL:
                model = self.load_mlflow_model()
            elif self.model_file_format == ModelFileFormat.PYTORCH_STATE_DICT:
                raise ValueError("Please use customized model loader to load state dict of model.")
            else:
                raise ValueError(f"Unsupported model file format: {self.model_file_format}")

        # we only have peft adapters for now
        adapter_path = self.get_resource("adapter_path")
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

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
            # default flavor is "hftransformersv2" from azureml.evaluate.mlflow>=0.0.8
            # "hftransformers" from azureml.evaluate.mlflow<0.0.8
            # TODO: let user specify flavor name if needed
            # TODO: to support other flavors in mlflow not only hftransformers
            hf_pretrained_class = None
            flavors = mlflow_data.get("flavors", {})
            if not flavors:
                raise ValueError(
                    "Invalid MLFlow model format. Please make sure the input model"
                    " format is same with the result of mlflow.transformers.save_model,"
                    " or aml_mlflow.hftransformers.save_model from azureml.evaluate.mlflow"
                )

            if "hftransformersv2" in flavors:
                hf_pretrained_class = flavors["hftransformersv2"].get("hf_pretrained_class", "AutoModel")
            elif "hftransformers" in flavors:
                hf_pretrained_class = flavors["hftransformers"].get("hf_pretrained_class", "AutoModel")
            else:
                raise ValueError(
                    "Unsupported MLFlow model flavor. Currently only support hftransformersv2/hftransformers."
                )

        model_loader = huggingface_model_loader(hf_pretrained_class)
        loaded_model = model_loader(tmp_dir_path)
        loaded_model.eval()

        tmp_dir.cleanup()

        return loaded_model

    def prepare_session(
        self,
        inference_settings: Dict[str, Any],
        device: Device,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        return self.load_model().eval()

    def get_dummy_inputs(self):
        """
        Return a dummy input for the model.
        """
        if self.dummy_inputs is not None:
            return self.dummy_inputs

        # Priority: dummy_inputs_func > io_config.input_shapes > hf_config.dataset > onnx_config
        dummy_inputs = None
        if self.dummy_inputs_func is not None:
            logger.debug("Using dummy_inputs_func to get dummy inputs")
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            dummy_inputs = user_module_loader.call_object(self.dummy_inputs_func, self)
        elif self.io_config and self.io_config["input_shapes"]:
            logger.debug("Using io_config.input_shapes to get dummy inputs")
            dummy_inputs, _ = (
                # input_types is optional
                data_config_template.dummy_data_config_template(
                    input_shapes=self.io_config["input_shapes"],
                    input_types=self.io_config.get("input_types"),
                )
                .to_data_container()
                .get_first_batch(data_root_path=None)
            )
        elif self.hf_config and self.hf_config.model_name and self.hf_config.task:
            if self.hf_config.dataset:
                logger.debug("Using hf_config.dataset to get dummy inputs")
                dummy_inputs, _ = (
                    data_config_template.huggingface_data_config_template(
                        self.hf_config.model_name,
                        self.hf_config.task,
                        **self.hf_config.dataset,
                    )
                    .to_data_container()
                    .get_first_batch(data_root_path=None)
                )
            elif not self.hf_config.components:
                logger.debug("Using hf onnx_config to get dummy inputs")
                dummy_inputs = get_hf_model_dummy_input(
                    self.hf_config.model_name, self.hf_config.task, self.hf_config.feature
                )

        if dummy_inputs is None:
            raise ValueError(
                "Unable to get dummy inputs. Please provide dummy_inputs_func, io_config.input_shapes,"
                " hf_config.dataset, or hf_config."
            )

        return dummy_inputs

    def get_hf_model_config(self):
        if self.hf_config is None:
            raise ValueError("HF model_config is not available")
        return self.hf_config.load_model_config(self.model_path)

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

        # get the component from hf_config
        components_dict = {component.name: component for component in self.hf_config.components}
        hf_component = components_dict[component_name]

        user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
        model_component = user_module_loader.call_object(hf_component.component_func, self.hf_config.model_name)

        io_config = hf_component.io_config
        if isinstance(io_config, str):
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            io_config = user_module_loader.call_object(hf_component.io_config, self.hf_config.model_name)
        io_config = validate_config(io_config, IOConfig)

        def model_loader(_):
            return model_component

        component_hf_config = deepcopy(self.hf_config).dict()
        component_hf_config.pop("components", None)

        return PyTorchModel(
            model_loader=model_loader,
            io_config=io_config,
            dummy_inputs_func=hf_component.dummy_inputs_func,
            model_script=self.model_script,
            script_dir=self.script_dir,
            hf_config=HFConfig.parse_obj(component_hf_config),
            model_attributes=self.model_attributes,
        )

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update(
            {
                "model_file_format": self.model_file_format,
                "model_loader": self.model_loader,
                "io_config": self.io_config,
                "dummy_inputs_func": self.dummy_inputs_func,
                "hf_config": self.hf_config,
            }
        )
        # convert script_dir and model_script to string
        # the original config has them as serialized ResourcePath
        for resource_name in ["script_dir", "model_script"]:
            if self.resource_paths[resource_name]:
                config["config"][resource_name] = self.get_resource(resource_name)
        return serialize_to_json(config, check_object)


class OptimumModel(PyTorchModel):
    def __init__(self, model_components: List[str], **kwargs):
        super().__init__(
            model_file_format=ModelFileFormat.OPTIMUM,
            **(kwargs or {}),
        )
        self.model_components = model_components

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update({"model_components": self.model_components})
        return serialize_to_json(config, check_object)


class SNPEModel(OliveModel):
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
        self,
        inference_settings: Dict[str, Any],
        device: Device,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ) -> SNPEInferenceSession:
        inference_settings = inference_settings or {}
        session_options = SNPESessionOptions(**inference_settings)
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
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_file_format: ModelFileFormat = ModelFileFormat.TENSORFLOW_SAVED_MODEL,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=model_path,
            framework=Framework.TENSORFLOW,
            model_file_format=model_file_format,
            model_attributes=model_attributes,
        )

    def load_model(self, rank: int = None):
        raise NotImplementedError()

    def prepare_session(
        self,
        inference_settings: Dict[str, Any],
        device: Device,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        raise NotImplementedError()


class OpenVINOModel(OliveModel):
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

    def load_model(self, rank: int = None):
        try:
            from openvino.tools.pot import load_model
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model")
        return load_model(self.model_config)

    def prepare_session(
        self,
        inference_settings: Dict[str, Any],
        device: Device,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
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
        model_filepaths: List[Union[Path, str]] = [],
        inference_settings: Optional[dict] = None,
        use_ort_extensions: bool = False,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_path=None,
            inference_settings=inference_settings,
            use_ort_extensions=use_ort_extensions,
            model_attributes=model_attributes,
        )
        self.model_filepaths = model_filepaths

    @property
    def ranks(self):
        return len(self.model_filepaths)

    def ranked_model_path(self, rank: int) -> Union[Path, str]:
        return self.model_filepaths[rank]

    def load_model(self, rank: int) -> ONNXModel:
        return ONNXModel(self.model_filepaths[rank], inference_settings=self.inference_settings)

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.GPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = 0,
    ):
        raise RuntimeError("DistributedOnnxModel doesn't have a session of its own")

    def get_default_execution_providers(self, filepath: str, device: Device):
        # return firstly available ep as ort default ep
        available_providers = DistributedOnnxModel.get_execution_providers(device)
        for ep in available_providers:
            if self._is_valid_ep(filepath, ep):
                return [ep]

        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def get_execution_providers(device: Device):
        import onnxruntime as ort

        eps_per_device = DistributedOnnxModel.EXECUTION_PROVIDERS.get(device)
        available_providers = ort.get_available_providers()
        return AcceleratorLookup.get_execution_providers(eps_per_device, available_providers)

    def to_json(self, check_object: bool = False):
        config = {
            "type": self.__class__.__name__,
            "config": {
                "model_filepaths": self.model_filepaths,
                "inference_settings": self.inference_settings,
                "use_ort_extensions": self.use_ort_extensions,
                "model_attributes": self.model_attributes,
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
        model_components: List[Union[ONNXModel, Dict[str, Any]]],
        model_component_names: List[str],
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_path=None, model_attributes=model_attributes)

        if isinstance(model_components[0], dict):
            assert all(
                [m.get("type").lower() == "onnxmodel" for m in model_components]
            ), "All components must be ONNXModel"
            self.model_components = [ONNXModel(**m.get("config", {})) for m in model_components]
        else:
            assert all([isinstance(m, ONNXModel) for m in model_components]), "All components must be ONNXModel"
            self.model_components = model_components

        assert len(self.model_components) == len(model_component_names), "Number of components and names must match"
        self.model_component_names = model_component_names

        for m in self.model_components:
            m.set_composite_parent(self)

    def load_model(self, rank: int = None):
        raise NotImplementedError()

    def prepare_session(
        self,
        inference_settings: Dict[str, Any],
        device: Device,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        raise NotImplementedError()

    def get_default_execution_providers(self, device: Device):
        raise NotImplementedError()

    def get_model_components(self):
        return self.model_components

    def get_model_component(self, idx):
        return self.model_components[idx]

    def get_model_component_names(self):
        return self.model_component_names

    def get_model_component_name(self, idx):
        return self.model_component_names[idx]

    def to_json(self, check_object: bool = False):
        json_dict = {
            "type": self.__class__.__name__,
            "config": {"model_attributes": self.model_attributes, "model_component_names": self.model_component_names},
        }
        json_dict["config"]["model_components"] = []
        for m in self.model_components:
            json_dict["config"]["model_components"].append(m.to_json(check_object))

        return serialize_to_json(json_dict, check_object)
