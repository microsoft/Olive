# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import yaml

from olive.common.config_utils import serialize_to_json, validate_config
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import copy_dir
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config import HfComponent, HfConfig, IoConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.model.handler.mixin import DummyInputsMixin, HfConfigMixin
from olive.model.utils.hf_utils import load_hf_model_from_model_class
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, ResourceType, create_resource_path

logger = logging.getLogger(__name__)


@model_handler_registry("PyTorchModel")
class PyTorchModelHandler(OliveModelHandler, HfConfigMixin, DummyInputsMixin):  # pylint: disable=too-many-ancestors
    """PyTorch model handler.

    Besides the model loading for PyTorch model, the model handler also provides the following functionalities:
      * Get the model io configuration either from user provider io_config or from hf_config. The priority is user
        provided io_config is higher than hf_config.
      * Get the dummy inputs for PyTorch model used to evaluate the latency.
      * All kinds of Hf model functionalities by HfConfigMixin.
    """

    resource_keys: Tuple[str, ...] = ("model_path", "script_dir", "model_script", "adapter_path")
    json_config_keys: Tuple[str, ...] = (
        "model_file_format",
        "model_loader",
        "io_config",
        "dummy_inputs_func",
        "hf_config",
    )

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_file_format: ModelFileFormat = ModelFileFormat.PYTORCH_ENTIRE_MODEL,
        model_loader: Union[str, Callable] = None,
        model_script: Union[str, Path] = None,
        script_dir: Union[str, Path] = None,
        io_config: Union[Dict[str, Any], IoConfig, str, Callable] = None,
        dummy_inputs_func: Union[str, Callable] = None,
        hf_config: Union[Dict[str, Any], HfConfig] = None,
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
        self.add_resources(locals())

        self.hf_config = None
        if hf_config:
            self.hf_config = validate_config(hf_config, HfConfig)
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
        self.io_config = (
            validate_config(io_config, IoConfig).dict() if isinstance(io_config, (IoConfig, dict)) else io_config
        )

        self.dummy_inputs_func = dummy_inputs_func

        self.dummy_inputs = None

    @property
    def script_dir(self) -> str:
        return self.get_resource("script_dir")

    @property
    def model_script(self) -> str:
        return self.get_resource("model_script")

    @property
    def adapter_path(self) -> str:
        return self.get_resource("adapter_path")

    def load_model(self, rank: int = None) -> torch.nn.Module:
        if self.model is not None:
            return self.model

        # Load user module at the beginning since we may need user defined models to load model
        user_module_loader = UserModuleLoader(self.model_script, self.script_dir)

        # Load special path or format model -> load model from hf config -> load normal path model
        if self.model_loader is not None:
            model = user_module_loader.call_object(self.model_loader, self.model_path)
        elif self.model_file_format == ModelFileFormat.PYTORCH_TORCH_SCRIPT:
            model = torch.jit.load(self.model_path)
        elif self.model_file_format == ModelFileFormat.PYTORCH_MLFLOW_MODEL:
            model = self._load_mlflow_model()
        elif self.hf_config and (self.hf_config.model_class or self.hf_config.task):
            model = self.load_hf_model(self.model_path)
        elif self.model_file_format == ModelFileFormat.PYTORCH_ENTIRE_MODEL:
            model = torch.load(self.model_path)
        elif self.model_file_format == ModelFileFormat.PYTORCH_STATE_DICT:
            raise ValueError("Please use customized model loader to load state dict of model.")
        else:
            raise ValueError(f"Unsupported model file format: {self.model_file_format}")

        # we only have peft adapters for now
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)

        self.model = model

        return model

    def get_component_model(self, component: HfComponent, rank: Optional[int] = None) -> "PyTorchModelHandler":
        if component.component_func is None:
            logger.debug("component_func is not provided, using hf_config to get component")
            model_component = self.load_hf_model(self.model_path)
        else:
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            model_component = user_module_loader.call_object(component.component_func, self)

        # the second default parameter is to fix ruff b023:
        # https://docs.astral.sh/ruff/rules/function-uses-loop-variable/
        def model_loader(_, model_component=model_component):
            return model_component

        component_hf_config = deepcopy(self.hf_config).dict()
        component_hf_config.pop("components", None)
        return PyTorchModelHandler(
            model_loader=model_loader,
            io_config=component.io_config,
            dummy_inputs_func=component.dummy_inputs_func,
            model_script=self.model_script,
            script_dir=self.script_dir,
            hf_config=HfConfig.parse_obj(component_hf_config),
            model_attributes=self.model_attributes,
        )

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        return self.load_model(rank).eval()

    def _load_mlflow_model(self):
        logger.info("Loading MLFlow model from %s", self.model_path)
        with tempfile.TemporaryDirectory(prefix="mlflow_tmp") as tmp_dir:
            copy_dir(os.path.join(self.model_path, "data/model"), tmp_dir, dirs_exist_ok=True)
            copy_dir(os.path.join(self.model_path, "data/config"), tmp_dir, dirs_exist_ok=True)
            copy_dir(os.path.join(self.model_path, "data/tokenizer"), tmp_dir, dirs_exist_ok=True)

            with open(os.path.join(self.model_path, "MLmodel")) as fp:
                mlflow_data = yaml.safe_load(fp)
                # default flavor is "hftransformersv2" from azureml.evaluate.mlflow>=0.0.8
                # "hftransformers" from azureml.evaluate.mlflow<0.0.8
                # TODO(trajep): let user specify flavor name if needed
                # to support other flavors in mlflow not only hftransformers
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
            loading_args = self.hf_config.get_loading_args_from_pretrained() if self.hf_config else {}
            loaded_model = load_hf_model_from_model_class(hf_pretrained_class, tmp_dir, **loading_args)
            loaded_model.eval()
            return loaded_model

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        # only keep model_attributes that are not in hf_config
        if self.model_attributes and self.hf_config:
            model_attributes = {}
            hf_config_dict = self.get_hf_model_config().to_dict()
            for key, value in self.model_attributes.items():
                if key not in hf_config_dict or hf_config_dict[key] != value:
                    model_attributes[key] = value
            config["config"]["model_attributes"] = model_attributes or None
        return serialize_to_json(config, check_object)

    def get_user_io_config(self, io_config: Union[Dict[str, Any], IoConfig, str, Callable]) -> Dict[str, Any]:
        """Resolve io_config to a dictionary.

        If io_config is a string name or a callable, it will be called to get io_config.
        """
        if isinstance(io_config, dict):
            # io_config is provided
            return io_config

        if isinstance(io_config, IoConfig):
            # io_config is an IoConfig
            return io_config.dict()

        if isinstance(io_config, (str, Callable)):
            # io_config is a string name or a callable
            logger.debug("Calling %s to get io_config", io_config)
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            io_config = user_module_loader.call_object(io_config, self)
            return validate_config(io_config, IoConfig).dict()

        return None

    def get_io_config(self) -> Dict[str, Any]:
        """Return io config of the model.

        Priority: io_config > hf_config (using onnx_config)
        """
        io_config = None
        if self.io_config:
            # io_config is provided
            io_config = self.get_user_io_config(self.io_config)
        elif self.hf_config and self.hf_config.task and not self.hf_config.components:
            # hf_config is provided
            logger.debug("Using hf onnx_config to get io_config")
            # For MLFlow model, get io config from model_name instead of model_path
            # TODO(xiaoyu): more investigation on the integration between MLFlow and HF
            io_config = self.get_hf_io_config()

        return io_config


@model_handler_registry("DistributedPyTorchModel")
class DistributedPyTorchModelHandler(OliveModelHandler, HfConfigMixin):
    resource_keys: Tuple[str, ...] = ("model_path", "script_dir", "model_script", "adapter_path")
    json_config_keys: Tuple[str, ...] = (
        "model_name_pattern",
        "num_ranks",
        "model_loader",
        "io_config",
        "dummy_inputs_func",
        "hf_config",
    )

    DEFAULT_RANKED_MODEL_NAME_FORMAT: ClassVar[str] = "model_{:02d}"

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
        model_name_pattern: str,
        num_ranks: int,
        model_file_format: ModelFileFormat = ModelFileFormat.PYTORCH_ENTIRE_MODEL,
        model_loader: Union[str, Callable] = None,
        model_script: Union[str, Path] = None,
        script_dir: Union[str, Path] = None,
        io_config: Union[Dict[str, Any], IoConfig, str, Callable] = None,
        dummy_inputs_func: Union[str, Callable] = None,
        hf_config: Union[Dict[str, Any], HfConfig] = None,
        adapter_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            framework=Framework.PYTORCH,
            model_file_format=model_file_format,
            model_path=model_path,
            model_attributes=model_attributes,
        )

        self.add_resources(locals())

        self.model_name_pattern = model_name_pattern
        self.num_ranks = num_ranks
        self.model_loader = model_loader
        self.io_config = (
            validate_config(io_config, IoConfig).dict() if isinstance(io_config, (IoConfig, dict)) else io_config
        )
        self.dummy_inputs_func = dummy_inputs_func
        self.hf_config = validate_config(hf_config, HfConfig) if hf_config else None

    @property
    def script_dir(self) -> str:
        return self.get_resource("script_dir")

    @property
    def model_script(self) -> str:
        return self.get_resource("model_script")

    @property
    def adapter_path(self) -> str:
        return self.get_resource("adapter_path")

    def ranked_model_name(self, rank: int) -> str:
        return self.model_name_pattern.format(rank)

    def ranked_model_path(self, rank: int) -> Union[Path, str]:
        return Path(self.model_path) / self.ranked_model_name(rank)

    def load_model(self, rank: int = None) -> PyTorchModelHandler:
        return PyTorchModelHandler(
            model_path=self.ranked_model_path(rank),
            model_file_format=ModelFileFormat.PYTORCH_ENTIRE_MODEL,
            model_loader=self.model_loader,
            model_script=self.model_script,
            script_dir=self.script_dir,
            io_config=self.io_config,
            dummy_inputs_func=self.dummy_inputs_func,
            hf_config=self.hf_config,
            adapter_path=self.adapter_path,
            model_attributes=self.model_attributes,
        )

    def get_component_model(self, component: HfComponent, rank: int = 0) -> PyTorchModelHandler:
        # TODO(shaahji): Add support for 'HfComponent.component_func'
        hf_config = deepcopy(self.hf_config).dict()
        hf_config.pop("components", None)
        return PyTorchModelHandler(
            model_path=self.ranked_model_path(rank),
            model_file_format=ModelFileFormat.PYTORCH_ENTIRE_MODEL,
            model_script=self.model_script,
            script_dir=self.script_dir,
            io_config=component.io_config,
            dummy_inputs_func=component.dummy_inputs_func,
            hf_config=HfConfig.parse_obj(hf_config),
            adapter_path=self.adapter_path,
            model_attributes=self.model_attributes,
        )

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.GPU,  # pylint: disable=signature-differs
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = 0,
    ) -> torch.nn.Module:
        return self.load_model(rank).load_model(rank).eval()
