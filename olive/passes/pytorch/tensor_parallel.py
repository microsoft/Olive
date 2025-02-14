# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Automatically distribute PyTorch model using Tensor Parallelism
# --------------------------------------------------------------------------

import logging
import multiprocessing
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Type

from olive.common.config_utils import ParamCategory
from olive.common.pydantic_v1 import validator
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import DistributedHfModelHandler, HfModelHandler
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pass_config import BasePassConfig
from olive.passes.pytorch.common import inherit_distributed_hf_from_hf

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class TensorParallel:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def replace_layers(self):
        raise NotImplementedError

    @abstractmethod
    def restore_layers(self):
        raise NotImplementedError

    @abstractmethod
    def split_weights(self, model: "torch.nn.Module"):
        raise NotImplementedError

    @abstractmethod
    def load_rank_weights(self, model: "torch.nn.Module"):
        raise NotImplementedError


class PyTorchTensorParallel(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        # Note : The default world_size should be the no of gpus (AcceleratorSpec.Device == GPU)
        # in the target OliveSystem
        return {
            "script_dir": PassConfigParam(
                type_=str,
                required=False,
                category=ParamCategory.PATH,
                description="Directory containing user script dependencies.",
            ),
            "user_script": PassConfigParam(
                type_=str,
                required=False,
                category=ParamCategory.PATH,
                description=(
                    "Path to user script. The values for other parameters which were assigned"
                    " function or object names will be imported from this script."
                ),
            ),
            "class_name": PassConfigParam(
                type_=str,
                required=True,
                category=ParamCategory.OBJECT,
                description="Class implementing model specific logic",
            ),
            "world_size": PassConfigParam(
                type_=int,
                default=2,
                required=True,
                description="Number of GPU nodes to distribute the model for. Must be greater than 1.",
            ),
            "parallel_jobs": PassConfigParam(
                type_=int,
                default=multiprocessing.cpu_count(),
                required=False,
                description="Number of parallel jobs. Defaulted to number of CPUs. Set it to 0 to disable.",
            ),
        }

    @staticmethod
    def _validate_world_size(v):
        if int(v) < 2:
            raise ValueError("world_size should be >= 2")

        return v

    @classmethod
    def _validators(cls) -> Dict[str, Callable]:
        return {"validate_distributor_config": validator("world_size", allow_reuse=True)(cls._validate_world_size)}

    @staticmethod
    def _generate_one(params):
        model_config, rank, world_size, output_filepath = params

        logger.debug("Exporting tensor parallel model for rank: %d, %s", rank, output_filepath)

        olive_model = HfModelHandler(**model_config)
        model_type = olive_model.get_hf_model_type()

        if model_type == "llama":
            from olive.passes.pytorch.tensor_parallel_llama2 import LlamaPyTorchTensorParallel

            impl = LlamaPyTorchTensorParallel(rank, world_size)
        else:
            raise ValueError("Unsupported model type '{model_type}' for tensor parallel pass")

        # 1. Replace the layers
        impl.replace_layers()

        try:
            # 2. Load the model
            pytorch_model = olive_model.load_model()
            pytorch_model.eval()
            pytorch_model.requires_grad_(False)
            pytorch_model.config.world_size = 1

            # 3. Split the weights
            impl.split_weights(pytorch_model)

            # 4. Load rank specific weights
            impl.load_rank_weights(pytorch_model)

            # 5. Save it out for each rank
            pytorch_model.config.world_size = world_size
            pytorch_model.save_pretrained(output_filepath)
            olive_model.save_metadata(output_filepath)
        finally:
            # 6. Restore layers that were replaced
            impl.restore_layers()

        logger.debug("Successfully exported tensor parallel model for rank: %d, %s", rank, output_filepath)

        return 1  # Return 1 for success.

    def _run_for_config(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> DistributedHfModelHandler:
        import torch

        world_size = int(config.world_size)
        output_model_path = Path(output_model_path)
        output_model_path.mkdir(parents=True, exist_ok=True)

        model_config = model.to_json()["config"]
        params = [
            (
                model_config,
                rank,
                world_size,
                output_model_path / DistributedHfModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT.format(rank),
            )
            for rank in range(world_size)
        ]

        max_parallel_jobs = min(world_size, config.parallel_jobs or multiprocessing.cpu_count())
        if max_parallel_jobs <= 1:
            results = [PyTorchTensorParallel._generate_one(_) for _ in params]
        else:
            with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
                results = pool.map(PyTorchTensorParallel._generate_one, params)

        if self.host_device == Device.GPU and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if world_size != sum(results):
            raise RuntimeError("Failed to create ranked tensor parallel models")

        # Finally, create DistributedHfModelHandler from ranked models for each rank
        return inherit_distributed_hf_from_hf(
            model, output_model_path, DistributedHfModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT, num_ranks=world_size
        )
