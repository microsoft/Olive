# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch

from olive.common.config_utils import serialize_to_json, validate_config
from olive.constants import Framework
from olive.model.config import HfLoadKwargs, IoConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.mixin import HfMixin, MLFlowMixin2
from olive.model.handler.pytorch2 import PyTorchModelHandlerBase
from olive.model.utils.hf_utils import load_hf_model_from_task
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)


@model_handler_registry("HFModel")
class HfModelHandler(PyTorchModelHandlerBase, MLFlowMixin2, HfMixin):  # pylint: disable=too-many-ancestors
    resource_keys: Tuple[str, ...] = ("model_path", "adapter_path")
    json_config_keys: Tuple[str, ...] = ("task", "load_kwargs", "generative")

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
        task: str,
        load_kwargs: Union[Dict[str, Any], HfLoadKwargs] = None,
        io_config: Union[Dict[str, Any], IoConfig, str] = None,
        adapter_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[Dict[str, Any]] = None,
        generative: bool = False,
    ):
        super().__init__(
            framework=Framework.PYTORCH,
            model_file_format=None,
            model_path=model_path,
            model_attributes=model_attributes,
            io_config=io_config,
            generative=generative,
        )
        self.add_resources(locals())
        self.task = task
        self.load_kwargs = validate_config(load_kwargs, HfLoadKwargs) if load_kwargs else None

        self.mlflow_model_path = None
        self.maybe_init_mlflow()

        self.model_attributes = {**self.get_hf_model_config().to_dict(), **(self.model_attributes or {})}

        self.model = None
        self.dummy_inputs = None

    @property
    def model_path(self):
        return self.mlflow_model_path or self.get_resource("model_path")

    @property
    def adapter_path(self) -> str:
        return self.get_resource("adapter_path")

    def load_model(self, rank: int = None) -> torch.nn.Module:
        """Load the model from the model path."""
        if self.model is not None:
            return self.model

        model = load_hf_model_from_task(self.task, self.model_path)

        # we only have peft adapters for now
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)

        self.model = model

        return model

    @property
    def io_config(self) -> Dict[str, Any]:
        """Return io config of the model.

        Priority: io_config > hf_config (using onnx_config)
        """
        io_config = None
        if self._io_config:
            # io_config is provided
            io_config = self.get_resolved_io_config(self._io_config, force_kv_cache=self.task.endswith("-with-past"))
        else:
            # hf_config is provided
            logger.debug("Trying hf onnx_config to get io_config")
            io_config = self.get_hf_io_config()
            if io_config:
                logger.debug("Got io_config from hf_config")

        return io_config

    def get_new_dummy_inputs(self):
        """Return a dummy input for the model."""
        # Priority: io_config > hf_config (using onnx_config)
        dummy_inputs = None

        dataloader = self._get_dummy_dataloader_from_io_config(force_kv_cache=self.task.endswith("-with-past"))
        if dataloader:
            dummy_inputs, _ = next(iter(dataloader))
        else:
            logger.debug("Trying hf onnx_config to get dummy inputs")
            dummy_inputs = self.get_hf_dummy_inputs()
            if dummy_inputs:
                logger.debug("Got dummy inputs from hf onnx_config")

        return dummy_inputs

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        # only keep model_attributes that are not in hf model config
        hf_config_dict = self.get_hf_model_config().to_dict()
        config["config"]["model_attributes"] = {
            key: value
            for key, value in self.model_attributes.items()
            if key not in hf_config_dict or hf_config_dict[key] != value
        } or None
        return serialize_to_json(config, check_object)
