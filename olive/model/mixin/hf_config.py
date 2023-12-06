# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.constants import ModelFileFormat
from olive.model.hf_utils import (
    get_hf_model_config,
    get_hf_model_io_config,
    load_huggingface_model_from_model_class,
    load_huggingface_model_from_task,
)

logger = logging.getLogger(__name__)


class HfConfigMixin:
    def get_hf_model_config(self):
        if self.hf_config is None:
            raise ValueError("HF model_config is not available")

        if self.model_file_format == ModelFileFormat.PYTORCH_MLFLOW_MODEL:
            return self._load_model_config(self.hf_config.model_path)
        else:
            return self._load_model_config(self.model_path)

    def load_hf_model(self, model_path: str = None):
        """Load model from model_path or model_name."""
        model_name_or_path = model_path or self.hf_config.model_name
        loading_args = self._get_loading_args()
        logger.info(f"Loading Huggingface model from {model_name_or_path}")
        if self.hf_config.task:
            model = load_huggingface_model_from_task(self.hf_config.task, model_name_or_path, **loading_args)
        elif self.hf_config.model_class:
            model = load_huggingface_model_from_model_class(
                self.hf_config.model_class, model_name_or_path, **loading_args
            )
        else:
            raise ValueError("Either task or model_class must be specified")

        return model

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

    def get_hf_io_config(self, model_path: str = None):
        """Get IO config for the model."""
        if self.hf_config and self.hf_config.task and not self.hf_config.components:
            hf_model_path = None if self.model_file_format == ModelFileFormat.PYTORCH_MLFLOW_MODEL else model_path
            return get_hf_model_io_config(
                hf_model_path or self.hf_config.model_name,
                self.hf_config.task,
                self.hf_config.feature,
                **self._get_loading_args(),
            )
        else:
            return None

    def _get_loading_args(self):
        return self.hf_config.from_pretrained_args.get_loading_args() if self.hf_config.from_pretrained_args else {}

    def _load_model_config(self, model_path: str = None):
        """Load model config from model_path or model_name."""
        return get_hf_model_config(model_path or self.hf_config.model_name, **self._get_loading_args())

    def get_hf_components(self):
        """Names of the components of the model."""
        if not self.hf_config or not self.hf_config.components:
            return None

        return [component.name for component in self.hf_config.components]
