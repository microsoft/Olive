# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from olive.common.config_utils import serialize_to_json, validate_config
from olive.common.user_module_loader import UserModuleLoader
from olive.constants import Framework, ModelFileFormat
from olive.hardware.accelerator import Device
from olive.model.config import IoConfig, complete_kv_cache_with_model_attributes, extend_io_config_with_kv_cache
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.model.handler.mixin import DummyInputsMixin, PytorchKvCacheMixin
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, ResourceType, create_resource_path

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class PyTorchModelHandlerBase(
    OliveModelHandler, DummyInputsMixin, PytorchKvCacheMixin
):  # pylint: disable=too-many-ancestors
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

    @staticmethod
    def get_resolved_io_config(
        io_config: Union[Dict[str, Any], IoConfig],
        force_kv_cache: bool = False,
        model_attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve io_config to a dictionary.

        :param io_config: io_config to resolve.
        :param force_kv_cache: whether to enable kv_cache if not already enabled.
        """
        io_config_obj = validate_config(io_config, IoConfig)

        # enable kv_cache
        io_config_obj.kv_cache = io_config_obj.kv_cache or force_kv_cache

        if io_config_obj.kv_cache:
            kv_cache_config = complete_kv_cache_with_model_attributes(io_config_obj.kv_cache, model_attributes or {})
            io_config_obj = extend_io_config_with_kv_cache(io_config_obj, kv_cache_config)
        return io_config_obj.dict(exclude_none=True)

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        # add _io_config to config to keep what was provided at init
        config["config"]["io_config"] = self._io_config
        return serialize_to_json(config, check_object)


@model_handler_registry("PyTorchModel")
class PyTorchModelHandler(PyTorchModelHandlerBase):  # pylint: disable=too-many-ancestors
    """PyTorch model handler.

    Besides the model loading for PyTorch model, the model handler also provides the following functionalities:
      * Get the model io configuration from user provider io_config.
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

        # ensure that script_dir and model_script are local resorces
        for resource_name, expected_type in [
            ("script_dir", ResourceType.LocalFolder),
            ("model_script", ResourceType.LocalFile),
        ]:
            resource = create_resource_path(self.get_resource(resource_name))
            if resource:
                assert resource.type == expected_type, f"{resource_name} must be a local {expected_type}."

        self.dummy_inputs_func = dummy_inputs_func
        self.dummy_inputs = None

    @property
    def script_dir(self) -> str:
        return self.get_resource("script_dir")

    @property
    def model_script(self) -> str:
        return self.get_resource("model_script")

    def load_model(self, rank: int = None, cache_model: bool = True) -> "torch.nn.Module":
        import torch

        if self.model:
            model = self.model
        else:
            # Load user module at the beginning since we may need user defined models to load model
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)

            # Load special path or format model -> load model from hf config -> load normal path model
            if self.model_loader is not None:
                model = user_module_loader.call_object(self.model_loader, self.model_path)
            elif self.model_file_format == ModelFileFormat.PYTORCH_TORCH_SCRIPT:
                model = torch.jit.load(self.model_path)
            elif self.model_file_format == ModelFileFormat.PYTORCH_ENTIRE_MODEL:
                model = torch.load(self.model_path, weights_only=False)
            elif self.model_file_format == ModelFileFormat.PYTORCH_SLICE_GPT_MODEL:
                model = self._load_slicegpt_model()
            elif self.model_file_format == ModelFileFormat.PYTORCH_STATE_DICT:
                raise ValueError("Please use customized model loader to load state dict of model.")
            else:
                raise ValueError(f"Unsupported model file format: {self.model_file_format}")

        self.model = model if cache_model else None

        return model

    def _load_slicegpt_model(self):
        from slicgpt.hf_utils import load_sliced_model

        model_name = self.model_attributes.get("model_name")
        if not model_name:
            raise ValueError("`model_name` model attribute is required to load SliceGPT model.")

        logger.info("Loading SliceGPT model with model_name %s from %s", model_name, self.model_path)
        loaded_model, _ = load_sliced_model(model_name, self.model_path)
        return loaded_model

    @property
    def io_config(self) -> Dict[str, Any]:
        """Return io config of the model."""
        if not self._io_config:
            return None

        io_config = self._io_config
        if isinstance(io_config, (str, Callable)):
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            io_config = user_module_loader.call_object(io_config, self)

        return self.get_resolved_io_config(io_config, model_attributes=self.model_attributes)

    def get_dummy_inputs(self, filter_hook=None, filter_hook_kwargs=None):
        """Return a dummy input for the model."""
        if self.dummy_inputs is not None:
            return self.dummy_inputs

        # Priority: user provided dummy_inputs_func > io_config
        if self.dummy_inputs_func is not None:
            logger.debug("Using dummy_inputs_func to get dummy inputs")
            user_module_loader = UserModuleLoader(self.model_script, self.script_dir)
            # respect user's dummy_inputs_func, no hook
            return user_module_loader.call_object(self.dummy_inputs_func, self)

        dummy_inputs = self._get_dummy_inputs_from_io_config(
            filter_hook=filter_hook, filter_hook_kwargs=filter_hook_kwargs
        )

        if dummy_inputs is None:
            raise ValueError("Unable to get dummy inputs for the model.")
        return dummy_inputs
