# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Automatically distribute PyTorch model using Tensor Parallelism
# --------------------------------------------------------------------------

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from pydantic import validator

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import DistributedPyTorchModel, PyTorchModel
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)


class PyTorchTensorParallel(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        # Note : The default world_size should be the no of gpus (AcceleratorSpec.Device == GPU)
        # in the target OliveSystem
        return {
            "world_size": PassConfigParam(
                type_=int,
                default=2,
                required=True,
                description=("Number of GPU nodes to distribute the model for. Must be greater than 1."),
            ),
        }

    @staticmethod
    def _validate_world_size(v):
        if int(v) < 2:
            raise ValueError("world_size should be >= 2")

        return v

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        return {
            "validate_distributor_config": validator("world_size", allow_reuse=True)(
                PyTorchTensorParallel._validate_world_size
            )
        }

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> DistributedPyTorchModel:
        world_size = int(config["world_size"])
        output_model_path = Path(output_model_path)

        # 1. Load the model
        pytorch_model = model.load_model()

        # 2. Replace the layers
        self.replace_layers()

        # 3. Split the weights
        self.split_weights(pytorch_model, world_size)

        try:
            # 4. Save the weights for each rank
            for rank in range(world_size):
                self.load_rank_weights(pytorch_model, rank, world_size)
                output_model_name = DistributedPyTorchModel.DEFAULT_RANKED_MODEL_NAME_FORMAT.format(rank)
                output_filepath = str(output_model_path / output_model_name)
                pytorch_model.save_pretrained(output_filepath)
        finally:
            # 5. Restore layers that were replaced
            self.restore_layers()

        # 6. Construct DistributedPyTorchModel from saved wegihts for each rank
        model_config = model.to_json()["config"]
        model_config["model_path"] = output_model_path
        model_config["model_name_pattern"] = DistributedPyTorchModel.DEFAULT_RANKED_MODEL_NAME_FORMAT
        model_config["num_ranks"] = world_size
        return DistributedPyTorchModel(**model_config)

    @abstractmethod
    def replace_layers(self):
        raise NotImplementedError

    @abstractmethod
    def restore_layers(self):
        raise NotImplementedError

    @abstractmethod
    def split_weights(self, model: torch.nn.Module, world_size: int):
        raise NotImplementedError

    @abstractmethod
    def load_rank_weights(self, model: torch.nn.Module, rank: int, world_size: int):
        raise NotImplementedError
