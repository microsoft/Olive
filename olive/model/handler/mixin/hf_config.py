# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple

from olive.common.user_module_loader import UserModuleLoader
from olive.constants import ModelFileFormat
from olive.model.config.hf_config import HfConfig
from olive.model.utils.hf_utils import (
    get_hf_model_config,
    get_hf_model_dummy_input,
    get_hf_model_io_config,
    load_huggingface_model_from_model_class,
    load_huggingface_model_from_task,
)

if TYPE_CHECKING:
    from olive.model.handler.pytorch import PyTorchModelHandler

logger = logging.getLogger(__name__)


class HfConfigMixin:
    """Provide the following Hugging Face model functionalites.

        * loading huggingface model
        * getting huggingface model config
        * getting huggingface model io config
        * getting huggingface model components like Whisper scenario.

    The mixin requires the following attributes to be set.
        * model_path
        * model_file_format
        * model_loader
        * model_script
        * script_dir
        * model_attributes
        * hf_config
    """

    def get_hf_model_config(self):
        if self.hf_config is None:
            raise ValueError("HF model_config is not available")

        return get_hf_model_config(self._get_model_path_or_name(), **HfConfigMixin._get_loading_args(self.hf_config))

    def get_hf_io_config(self):
        """Get Io config for the model."""
        if self.hf_config and self.hf_config.task and not self.hf_config.components:
            return get_hf_model_io_config(
                self._get_model_path_or_name(),
                self.hf_config.task,
                self.hf_config.feature,
                **HfConfigMixin._get_loading_args(self.hf_config),
            )
        else:
            return None

    def get_hf_components(self) -> List[Tuple[str, "PyTorchModelHandler"]]:
        # the following import is to solve circular import
        from olive.model.handler.pytorch import PyTorchModelHandler

        if not self.hf_config or not self.hf_config.components:
            return

        for component in self.hf_config.components:
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
            yield component.name, PyTorchModelHandler(
                model_loader=model_loader,
                io_config=component.io_config,
                dummy_inputs_func=component.dummy_inputs_func,
                model_script=self.model_script,
                script_dir=self.script_dir,
                hf_config=HfConfig.parse_obj(component_hf_config),
                model_attributes=self.model_attributes,
            )

    @staticmethod
    def load_hf_model(hf_config, model_path: str = None):
        """Load model from model_path or model_name."""
        model_name_or_path = model_path or hf_config.model_name
        loading_args = HfConfigMixin._get_loading_args(hf_config)
        logger.info(f"Loading Huggingface model from {model_name_or_path}")
        if hf_config.task:
            model = load_huggingface_model_from_task(hf_config.task, model_name_or_path, **loading_args)
        elif hf_config.model_class:
            model = load_huggingface_model_from_model_class(hf_config.model_class, model_name_or_path, **loading_args)
        else:
            raise ValueError("Either task or model_class must be specified")

        return model

    def get_hf_dummy_inputs(self):
        """Get dummy inputs for the model."""
        return get_hf_model_dummy_input(
            self.model_path or self.hf_config.model_name,
            self.hf_config.task,
            self.hf_config.feature,
            **HfConfigMixin._get_loading_args(self.hf_config),
        )

    def is_model_loaded_from_hf_config(self) -> bool:
        """Return True if the model is loaded from hf_config, False otherwise."""
        return (
            (not self.model_loader)
            and (
                self.model_file_format
                not in (ModelFileFormat.PYTORCH_TORCH_SCRIPT, ModelFileFormat.PYTORCH_MLFLOW_MODEL)
            )
            and self.hf_config
            and (self.hf_config.model_class or self.hf_config.task)
        )

    @staticmethod
    def _get_loading_args(hf_config):
        return hf_config.from_pretrained_args.get_loading_args() if hf_config.from_pretrained_args else {}

    def _get_model_path_or_name(self):
        if self.model_file_format == ModelFileFormat.PYTORCH_MLFLOW_MODEL:
            return self.hf_config.model_name
        else:
            return self.model_path or self.hf_config.model_name
