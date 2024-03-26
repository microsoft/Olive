# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Automatically distribute PyTorch model using Pipeline Parallelism
# --------------------------------------------------------------------------

import logging
import multiprocessing
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict

import torch

from olive.common.pydantic_v1 import validator
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import DistributedPyTorchModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)


class PipelineParallel:
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
    def split_layers(self, model: torch.nn.Module):
        raise NotImplementedError


class PyTorchPipelineParallel(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        # Note : The default value for nstages should be the no of gpus
        # (AcceleratorSpec.Device == GPU) in the target OliveSystem
        return {
            "nstages": PassConfigParam(
                type_=int,
                default=2,
                required=True,
                description="Number of stages to distribute the model for. Must be greater than 1.",
            ),
            "parallel_jobs": PassConfigParam(
                type_=int,
                default=multiprocessing.cpu_count(),
                required=False,
                description="Number of parallel jobs. Defaulted to number of CPUs. Set it to 0 to disable.",
            ),
        }

    @staticmethod
    def _validate_nstages(v):
        if int(v) < 2:
            raise ValueError("nstages should be >= 2")

        return v

    @classmethod
    def _validators(cls) -> Dict[str, Callable]:
        return {"validate_distributor_config": validator("nstages", allow_reuse=True)(cls._validate_nstages)}

    @staticmethod
    def _generate_one(params):
        model_config, rank, world_size, output_filepath = params

        logger.debug("Exporting pipeline parallel model for rank: %d, %s", rank, output_filepath)

        from olive.passes.pytorch.pipeline_parallel_transformer import TransformerPipelineParallel

        impl = TransformerPipelineParallel(rank, world_size)

        # 1. Replace the layers
        impl.replace_layers()

        try:
            # 2. Load the model
            olive_model = PyTorchModelHandler(**model_config)
            pytorch_model = olive_model.load_model()
            pytorch_model.eval()
            pytorch_model.requires_grad_(False)
            pytorch_model.config.world_size = 1

            # 3. Split the layers
            impl.split_layers(pytorch_model)

            # 4. Save it out for each rank
            pytorch_model.config.rank = rank
            pytorch_model.config.world_size = world_size
            torch.save(pytorch_model.state_dict(), output_filepath)
        finally:
            # 5. Restore layers that were replaced
            impl.restore_layers()

        logger.debug("Successfully exported pipeline parallel model for rank: %d, %s", rank, output_filepath)

        return 1  # Return 1 for success.

    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> DistributedPyTorchModelHandler:
        world_size = int(config["nstages"])
        output_model_path = Path(output_model_path)
        output_model_path.mkdir(parents=True, exist_ok=True)

        ranked_model_name_format = DistributedPyTorchModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT + ".pt"

        model_config = model.to_json()["config"]
        params = [
            (
                model_config,
                rank,
                world_size,
                output_model_path / ranked_model_name_format.format(rank),
            )
            for rank in range(world_size)
        ]

        max_parallel_jobs = min(world_size, config["parallel_jobs"] or multiprocessing.cpu_count())
        if max_parallel_jobs <= 1:
            results = [PyTorchPipelineParallel._generate_one(_) for _ in params]
        else:
            with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
                results = pool.map(PyTorchPipelineParallel._generate_one, params)

        if self.accelerator_spec.accelerator_type == Device.GPU and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if world_size != sum(results):
            raise RuntimeError("Failed to create ranked pipeline parallel models")

        # Finally, create DistributedPyTorchModel from ranked models for each rank
        model_config = model.to_json()["config"]
        del model_config["model_loader"]
        model_config["model_path"] = output_model_path
        model_config["model_name_pattern"] = DistributedPyTorchModelHandler.DEFAULT_RANKED_MODEL_NAME_FORMAT
        model_config["num_ranks"] = world_size
        return DistributedPyTorchModelHandler(**model_config)
