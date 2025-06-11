# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

from olive.common.config_utils import serialize_to_json, validate_config
from olive.common.constants import DEFAULT_HF_TASK
from olive.common.hf.utils import load_model_from_task
from olive.common.utils import dict_diff
from olive.constants import Framework
from olive.hardware.accelerator import Device
from olive.model.config import HfLoadKwargs, IoConfig
from olive.model.config.registry import model_handler_registry
from olive.model.handler.base import OliveModelHandler
from olive.model.handler.mixin import HfMixin, MLFlowTransformersMixin
from olive.model.handler.pytorch import PyTorchModelHandlerBase
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@model_handler_registry("HFModel")
class HfModelHandler(PyTorchModelHandlerBase, MLFlowTransformersMixin, HfMixin):  # pylint: disable=too-many-ancestors
    resource_keys: tuple[str, ...] = ("model_path", "adapter_path")
    json_config_keys: tuple[str, ...] = ("task", "load_kwargs")

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
        task: str = DEFAULT_HF_TASK,
        load_kwargs: Union[dict[str, Any], HfLoadKwargs] = None,
        io_config: Union[dict[str, Any], IoConfig, str] = None,
        adapter_path: OLIVE_RESOURCE_ANNOTATIONS = None,
        model_attributes: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            model_file_format=None,
            model_path=model_path,
            model_attributes=model_attributes,
            io_config=io_config,
        )
        self.add_resources(locals())
        self.task = task
        self.load_kwargs = validate_config(load_kwargs, HfLoadKwargs, warn_unused_keys=False) if load_kwargs else None

        self.model_attributes = {**self.get_hf_model_config().to_dict(), **(self.model_attributes or {})}

        self.model = None
        self.dummy_inputs = None

    @property
    def model_name_or_path(self) -> str:
        """Return the path to valid hf transformers checkpoint.

        Call this instead of model_path if you expect a checkpoint path.
        """
        return self.get_mlflow_transformers_path() or self.model_path

    @property
    def adapter_path(self) -> str:
        """Return the path to the peft adapter."""
        return self.get_resource("adapter_path")

    def load_model(self, rank: int = None, cache_model: bool = True) -> "torch.nn.Module":
        """Load the model from the model path."""
        if self.model:
            model = self.model
        else:
            model = load_model_from_task(self.task, self.model_path, **self.get_load_kwargs())

            # we only have peft adapters for now
            if self.adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path)

        self.model = model if cache_model else None

        return model

    @property
    def io_config(self) -> dict[str, Any]:
        """Return io config of the model.

        Priority: io_config > hf onnx_config
        """
        io_config = None
        if self._io_config:
            # io_config is provided
            io_config = self.get_resolved_io_config(
                self._io_config, force_kv_cache=self.task.endswith("-with-past"), model_attributes=self.model_attributes
            )
        else:
            logger.debug("Trying hf optimum export config to get io_config")
            io_config = self.get_hf_io_config()
            if io_config:
                logger.debug("Got io_config from hf optimum export config")

        return io_config

    def get_dummy_inputs(self, filter_hook=None, filter_hook_kwargs=None):
        """Return a dummy input for the model."""
        if self.dummy_inputs is not None:
            return self.dummy_inputs

        # Priority: io_config > hf onnx_config
        dummy_inputs = self._get_dummy_inputs_from_io_config(
            filter_hook=filter_hook,
            filter_hook_kwargs=filter_hook_kwargs,
        )
        if dummy_inputs is not None:
            return dummy_inputs

        logger.debug("Trying hf optimum export config to get dummy inputs")
        dummy_inputs = self.get_hf_dummy_inputs()
        if dummy_inputs is not None:
            logger.debug("Got dummy inputs from hf optimum export config")
        else:
            raise ValueError(
                "Unable to get dummy inputs for the model. Please provide io_config or install an optimum version that"
                " supports the model for export."
            )

        return dummy_inputs

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        # only keep model_attributes that are not in hf model config
        hf_model_config_dict = self.get_hf_model_config().to_dict()
        config["config"]["model_attributes"] = dict_diff(self.model_attributes, hf_model_config_dict)
        return serialize_to_json(config, check_object)


@model_handler_registry("DistributedHfModel")
class DistributedHfModelHandler(OliveModelHandler):
    json_config_keys: tuple[str, ...] = (
        "model_name_pattern",
        "num_ranks",
        "task",
        "load_kwargs",
        "io_config",
    )

    DEFAULT_RANKED_MODEL_NAME_FORMAT: ClassVar[str] = "model_{:02d}"

    def __init__(
        self,
        model_path: OLIVE_RESOURCE_ANNOTATIONS,
        model_name_pattern: str,
        num_ranks: int,
        task: str = DEFAULT_HF_TASK,
        load_kwargs: Union[dict[str, Any], HfLoadKwargs] = None,
        io_config: Union[dict[str, Any], IoConfig] = None,
        model_attributes: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            framework=Framework.PYTORCH,
            model_file_format=None,
            model_path=model_path,
            model_attributes=model_attributes,
            io_config=io_config,
        )
        self.add_resources(locals())

        self.model_name_pattern = model_name_pattern
        self.num_ranks = num_ranks
        self.task = task
        self.load_kwargs = load_kwargs

    def ranked_model_name(self, rank: int) -> str:
        return self.model_name_pattern.format(rank)

    def ranked_model_path(self, rank: int) -> Union[Path, str]:
        return Path(self.model_path) / self.ranked_model_name(rank)

    def load_model(self, rank: int = None, cache_model: bool = True) -> HfModelHandler:
        return HfModelHandler(
            model_path=self.ranked_model_path(rank),
            task=self.task,
            load_kwargs=self.load_kwargs,
            io_config=self.io_config,
            model_attributes=self.model_attributes,
        )

    def prepare_session(
        self,
        inference_settings: Optional[dict[str, Any]] = None,
        device: Device = Device.GPU,  # pylint: disable=signature-differs
        execution_providers: Union[str, list[str]] = None,
        rank: Optional[int] = 0,
    ) -> "torch.nn.Module":
        return self.load_model(rank).load_model(rank).eval()

    def run_session(
        self,
        session: Any = None,
        inputs: Union[dict[str, Any], list[Any], tuple[Any, ...]] = None,
        **kwargs: dict[str, Any],
    ) -> Any:
        is_generative = False
        if self.get_hf_generation_config() is not None:
            is_generative = True
        if isinstance(inputs, dict):
            results = session.generate(**inputs, **kwargs) if is_generative else session(**inputs, **kwargs)
        else:
            results = session.generate(inputs, **kwargs) if is_generative else session(inputs, **kwargs)
        return results
