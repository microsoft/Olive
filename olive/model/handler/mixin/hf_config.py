# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import TYPE_CHECKING, Generator, Optional, Tuple

from olive.constants import ModelFileFormat
from olive.model.utils.hf_utils import (
    get_hf_model_config,
    get_hf_model_dummy_input,
    get_hf_model_io_config,
    load_hf_model_from_model_class,
    load_hf_model_from_task,
)

if TYPE_CHECKING:
    from olive.model.handler.pytorch import PyTorchModelHandler

logger = logging.getLogger(__name__)


class HfConfigMixin:
    """Provide the following Hugging Face model functionalities.

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

        return get_hf_model_config(self._get_model_path_or_name(), **self.hf_config.get_loading_args_from_pretrained())

    def get_hf_io_config(self):
        """Get Io config for the model."""
        if self.hf_config and self.hf_config.task and not self.hf_config.components:
            return get_hf_model_io_config(
                self._get_model_path_or_name(),
                self.hf_config.task,
                self.hf_config.feature,
                **self.hf_config.get_loading_args_from_pretrained(),
            )
        else:
            return None

    def get_hf_components(self, rank: Optional[int] = None) -> Generator[Tuple[str, "PyTorchModelHandler"], None, None]:
        if self.hf_config and self.hf_config.components:
            for component in self.hf_config.components:
                yield component.name, self.get_component_model(component, rank)

    def load_hf_model(self, model_path: str = None):
        """Load model from model_path or model_name."""
        model_name_or_path = model_path or self.hf_config.model_name
        loading_args = self.hf_config.get_loading_args_from_pretrained()
        logger.info("Loading Huggingface model from %s", model_name_or_path)
        if self.hf_config.task:
            model = load_hf_model_from_task(self.hf_config.task, model_name_or_path, **loading_args)
        elif self.hf_config.model_class:
            model = load_hf_model_from_model_class(self.hf_config.model_class, model_name_or_path, **loading_args)
        else:
            raise ValueError("Either task or model_class must be specified")

        return model

    def get_hf_dummy_inputs(self):
        """Get dummy inputs for the model."""
        return get_hf_model_dummy_input(
            self._get_model_path_or_name(),
            self.hf_config.task,
            self.hf_config.feature,
            **self.hf_config.get_loading_args_from_pretrained(),
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

    def _get_model_path_or_name(self):
        if self.model_file_format == ModelFileFormat.PYTORCH_MLFLOW_MODEL:
            return self.hf_config.model_name
        else:
            return self.model_path or self.hf_config.model_name
