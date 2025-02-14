# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# -------------------------------------------------------------------------
import json
import logging
import sys
from typing import Dict, Type, Union

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from olive.common.config_utils import validate_config
from olive.constants import ModelFileFormat
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pass_config import BasePassConfig
from olive.passes.pytorch.common import inherit_pytorch_from_hf

logger = logging.getLogger(__name__)


class SliceGPT(Pass):
    """Run SliceGPT on a Hugging Face PyTorch model.

    See https://arxiv.org/pdf/2401.15024.pdf for more details on the algorithm.

    This pass only supports HfModelHandler.
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "calibration_data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                required=True,
                description="Data config for Dataset to calibrate and calculate perplexity on.",
            ),
            "calibration_nsamples": PassConfigParam(
                type_=int,
                required=False,
                default_value=128,
                description="Number of samples of the calibration data to load.",
            ),
            "calibration_batch_size": PassConfigParam(
                type_=int,
                required=False,
                default_value=16,
                description="Batch size for loading the calibration data.",
            ),
            "seed": PassConfigParam(
                type_=int,
                required=False,
                default_value=42,
                description="Seed for sampling the calibration data.",
            ),
            "sparsity": PassConfigParam(
                type_=float,
                default_value=0.0,
                description="A measure of how much slicing is applied (in the range [0, 1))",
            ),
            "round_interval": PassConfigParam(
                type_=int,
                default_value=8,
                description="Interval for rounding the weights (the best value may depend on your hardware)",
            ),
            "final_orientation": PassConfigParam(
                type_=str,
                default_value="random",
                description="Final orientation of the sliced weights. Choices are random or pca.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> PyTorchModelHandler:
        if sys.version_info < (3, 10):
            raise ValueError("SliceGPT requires python3.10 or higher")

        from slicegpt import layernorm_fusion, rotate
        from slicegpt.hf_utils import get_model_and_tokenizer
        from slicegpt.slicing_scheduler import ConstSlicingScheduler

        # Renaming variables to match their contextual use
        model_handler = model
        model = None

        model_adapter, _ = get_model_and_tokenizer(model_handler.model_name_or_path)
        model_handler.model = model_adapter.model
        model = model_handler.load_model()

        # replace and fuse layers
        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)

        original_param_count = sum(int(p.nelement()) for p in model.parameters())
        logger.info("Original model parameters: %s", f"{original_param_count:,}")

        # compute new embedding dimension given the desired sparsity level
        new_embedding_dim = int((1 - config.sparsity) * model_adapter.hidden_size)
        # round (down) to the nearest multiple of round_interval
        new_embedding_dim -= new_embedding_dim % config.round_interval
        logger.info(
            "New embedding dimension: %f (sparsity %.4f%%)",
            new_embedding_dim,
            100 * (1 - new_embedding_dim / model_adapter.hidden_size),
        )

        data_config = validate_config(config.calibration_data_config, DataConfig)
        dataloader = data_config.to_data_container().create_dataloader()
        dataset = [
            {
                "input_ids": data[0]["input_ids"].squeeze(),
                "attention_mask": data[0]["attention_mask"].squeeze(),
                "labels": data[1].squeeze(),
            }
            for data in dataloader
        ]

        torch.manual_seed(config.seed)
        sampler = SubsetRandomSampler(torch.randperm(len(dataset))[: config.calibration_nsamples])
        train_loader = DataLoader(dataset, batch_size=config.calibration_batch_size, sampler=sampler)

        # rotate and slice
        schedular = ConstSlicingScheduler(new_embedding_dim)
        rotate.rotate_and_slice(model_adapter, train_loader, schedular, final_orientation=config.final_orientation)

        sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
        sliced_fraction = 1.0 - sliced_param_count / original_param_count
        logger.info("Sliced model parameters: %s (sliced fraction %.4f)", f"{sliced_param_count:,}", sliced_fraction)

        output_model_filepath = normalize_path_suffix(output_model_path, "model.pt")
        torch.save(model.state_dict(), output_model_filepath)

        output_config_filepath = normalize_path_suffix(output_model_path, "config.json")
        with open(output_config_filepath, "w") as strm:
            json.dump(model_handler.get_hf_model_config().to_dict(), strm, indent=4)

        output_slice_config_filepath = normalize_path_suffix(output_model_path, "model.json")
        with open(output_slice_config_filepath, "w") as strm:
            json.dump(model_adapter.slicing_conf.to_dict(), strm, indent=4)

        # return PyTorchModelHandler
        return inherit_pytorch_from_hf(
            model_handler,
            output_model_path,
            model_file_format=ModelFileFormat.PYTORCH_SLICE_GPT_MODEL,
            model_name=model_handler.model_name_or_path,
        )
