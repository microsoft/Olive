# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import torch

from olive.common.config_utils import serialize_to_json
from olive.common.user_module_loader import UserModuleLoader
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config import (
    HfComponent,
    IoConfig,
    complete_kv_cache_with_model_attributes,
    extend_io_config_with_kv_cache,
)
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.model.handler.mixin import DummyInputsMixin, HfConfigMixin, MLFlowMixin, PytorchKvCacheMixin
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, ResourceType, create_resource_path

logger = logging.getLogger(__name__)


class PyTorchModelHandlerBase(OliveModelHandler, DummyInputsMixin):
    """Base class for PyTorch model handler."""

    def prepare_session(
        self,
        inference_settings: Optional[Dict[str, Any]] = None,
        device: Device = Device.CPU,
        execution_providers: Union[str, List[str]] = None,
        rank: Optional[int] = None,
    ):
        return self.load_model(rank).eval()

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        if isinstance(inputs, dict):
            results = session.generate(**inputs, **kwargs) if self.generative else session(**inputs, **kwargs)
        else:
            results = session.generate(inputs, **kwargs) if self.generative else session(inputs, **kwargs)
        return results

    def get_resolved_io_config(
        self,
        io_config: Union[Dict[str, Any], IoConfig, str, Callable],
        force_kv_cache: bool = False,
    ) -> Dict[str, Any]:
        """Resolve io_config to a dictionary.

        :param io_config: io_config to resolve.
        :param force_kv_cache: whether to enable kv_cache if not already enabled.
        """
        io_config_obj = None
        if isinstance(io_config, dict):
            io_config_obj = IoConfig.parse_obj(io_config)
        elif isinstance(io_config, IoConfig):
            # return a new copy of io_config to avoid modifying the original one
            io_config_obj = io_config.copy(deep=True)
        else:
            raise ValueError(f"Unsupported io_config type: {type(io_config)}")

        # enable kv_cache
        io_config_obj.kv_cache = io_config_obj.kv_cache or force_kv_cache

        if io_config_obj.kv_cache:
            kv_cache_config = complete_kv_cache_with_model_attributes(io_config_obj.kv_cache, self.model_attributes)
            io_config_obj = extend_io_config_with_kv_cache(io_config_obj, kv_cache_config)
        return io_config_obj.dict(exclude_none=True)

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        # add _io_config to config to keep what was provided at init
        config["config"]["io_config"] = self._io_config
        return serialize_to_json(config, check_object)


@model_handler_registry("PyTorchModel2")
class PyTorchModelHandler2(
    PyTorchModelHandlerBase, PytorchKvCacheMixin, MLFlowMixin
):  # pylint: disable=too-many-ancestors
    """PyTorch model handler.

    Besides the model loading for PyTorch model, the model handler also provides the following functionalities:
      * Get the model io configuration either from user provider io_config or from hf_config. The priority is user
        provided io_config is higher than hf_config.
      * Get the dummy inputs for PyTorch model used to evaluate the latency.
    """

    resource_keys: Tuple[str, ...] = ("model_path", "script_dir", "model_script")
    json_config_keys: Tuple[str, ...] = ("model_file_format", "model_loader", "dummy_inputs_func", "generative")

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_file_format: ModelFileFormat = ModelFileFormat.PYTORCH_ENTIRE_MODEL,
        model_loader: Union[str, Callable] = None,
        model_script: Union[str, Path] = None,
        script_dir: Union[str, Path] = None,
        io_config: Union[Dict[str, Any], IoConfig, str, Callable] = None,
        dummy_inputs_func: Union[str, Callable] = None,
        model_attributes: Optional[Dict[str, Any]] = None,
        generative: bool = False,
    ):
        if not (isinstance(model_loader, Callable) or (isinstance(model_loader, str) and model_script) or model_path):
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
            io_config=io_config,
            generative=generative,
        )
        self.add_resources(locals())

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

        # Load user module at the beginning since we may need user defined models to load model
        user_module_loader = UserModuleLoader(self.model_script, self.script_dir)

        # Load special path or format model -> load model from hf config -> load normal path model
        if self.model_loader is not None:
            model = user_module_loader.call_object(self.model_loader, self.model_path)
        elif self.model_file_format == ModelFileFormat.PYTORCH_TORCH_SCRIPT:
            model = torch.jit.load(self.model_path)
        elif self.model_file_format == ModelFileFormat.PYTORCH_ENTIRE_MODEL:
            model = torch.load(self.model_path)
        elif self.model_file_format == ModelFileFormat.PYTORCH_SLICE_GPT_MODEL:
            model = self._load_slicegpt_model()
        elif self.model_file_format == ModelFileFormat.PYTORCH_STATE_DICT:
            raise ValueError("Please use customized model loader to load state dict of model.")
        else:
            raise ValueError(f"Unsupported model file format: {self.model_file_format}")

        self.model = model

        return model

    def _load_slicegpt_model(self):
        from slicgpt.hf_utils import load_sliced_model

        model_name = self.model_attributes.get("model_name")
        if not model_name:
            raise ValueError("`model_name` model attributed is required to load SliceGPT model.")

        logger.info("Loading SliceGPT model with model_name %s from %s", model_name, self.model_path)
        loaded_model, _ = load_sliced_model(model_name, self.model_path)
        return loaded_model

    @property
    def io_config(self) -> Dict[str, Any]:
        """Return io config of the model.

        Priority: io_config > hf_config (using onnx_config)
        """
        if not self._io_config:
            return None

        return self.get_resolved_io_config(self._io_config)

    def get_new_dummy_inputs(self):
        """Return a dummy input for the model."""
        # Priority: user provided dummy_inputs_func > io_config
        dummy_inputs = None

        if self.dummy_inputs_func is not None:
            logger.debug("Using dummy_inputs_func to get dummy inputs")
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            dummy_inputs = user_module_loader.call_object(self.dummy_inputs_func, self)
            # respect user's dummy_inputs_func, no hook
        else:
            dataloader = self._get_dummy_dataloader_from_io_config()
            if dataloader:
                dummy_inputs, _ = dataloader.get_first_batch()

        return dummy_inputs


@model_handler_registry("DistributedPyTorchModel2")
class DistributedPyTorchModelHandler2(OliveModelHandler, HfConfigMixin):
    resource_keys: Tuple[str, ...] = ("model_path", "script_dir", "model_script")
    json_config_keys: Tuple[str, ...] = (
        "model_name_pattern",
        "num_ranks",
        "model_loader",
        "io_config",
        "dummy_inputs_func",
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
        model_attributes: Optional[Dict[str, Any]] = None,
        generative: bool = False,
    ):
        super().__init__(
            framework=Framework.PYTORCH,
            model_file_format=model_file_format,
            model_path=model_path,
            model_attributes=model_attributes,
            io_config=io_config,
            generative=generative,
        )

        self.add_resources(locals())

        self.model_name_pattern = model_name_pattern
        self.num_ranks = num_ranks
        self.model_loader = model_loader
        self.dummy_inputs_func = dummy_inputs_func

    @property
    def script_dir(self) -> str:
        return self.get_resource("script_dir")

    @property
    def model_script(self) -> str:
        return self.get_resource("model_script")

    def ranked_model_name(self, rank: int) -> str:
        return self.model_name_pattern.format(rank)

    def ranked_model_path(self, rank: int) -> Union[Path, str]:
        return Path(self.model_path) / self.ranked_model_name(rank)

    def load_model(self, rank: int = None) -> PyTorchModelHandler2:
        return PyTorchModelHandler2(
            model_path=self.ranked_model_path(rank),
            model_file_format=ModelFileFormat.PYTORCH_ENTIRE_MODEL,
            model_loader=self.model_loader,
            model_script=self.model_script,
            script_dir=self.script_dir,
            io_config=self._io_config,
            dummy_inputs_func=self.dummy_inputs_func,
            model_attributes=self.model_attributes,
        )

    def get_component_model(self, component: HfComponent, rank: int = 0) -> PyTorchModelHandler2:
        return PyTorchModelHandler2(
            model_path=self.ranked_model_path(rank),
            model_file_format=ModelFileFormat.PYTORCH_ENTIRE_MODEL,
            model_script=self.model_script,
            script_dir=self.script_dir,
            io_config=component.io_config,
            dummy_inputs_func=component.dummy_inputs_func,
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

    def run_session(
        self,
        session: Any = None,
        inputs: Union[Dict[str, Any], List[Any], Tuple[Any, ...]] = None,
        **kwargs: Dict[str, Any],
    ) -> Any:
        if isinstance(inputs, dict):
            results = session.generate(**inputs, **kwargs) if self.generative else session(**inputs, **kwargs)
        else:
            results = session.generate(inputs, **kwargs) if self.generative else session(inputs, **kwargs)
        return results
