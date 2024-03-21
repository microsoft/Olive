# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# -------------------------------------------------------------------------
import logging
from typing import Any, Dict, List, Union

import torch

from olive.common.config_utils import ConfigBase
from olive.constants import ModelFileFormat
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModelHandler
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)

class SliceGPT(Pass):
    """Run SliceGPT on a Hugging Face PyTorch model.

    See https://arxiv.org/pdf/2401.15024.pdf for more details on the algorithm.

    This pass only supports PyTorchModelHandler with hf_config. 
    """
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "sparsity": PassConfigParam(
                type_=float, default_value=0.0, 
                description="A measure of how much slicing is applied (in the range [0, 1))"
            ),
            "final_orientation": PassConfigParam(
                type_=str,
                default_value="random",
                description="Final orientation of the sliced weights. Choices are random or pca."
            ),
            "round_interval": PassConfigParam(
                type_=int,
                default_value=8,
                description="Interval for rounding the weights (the best value may depend on your hardware)",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                required=True,
                description=(
                    "Data config to use for SliceGPT weights."
                ),
            ),
        }

    @staticmethod
    def get_datasets(config: ConfigBase, data_root: str):
        """Load training and evaluation datasets."""
        train_data_config = config.data_config

        # load training dataset
        train_data_container = train_data_config.to_data_container()
        train_dataset = train_data_container.pre_process(train_data_container.load_dataset(data_root))
        train_dataset = train_dataset.to_hf_dataset(label_name="labels")

        return train_dataset

    @torch.no_grad()
    def _run_for_config(
        self, model_handler: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:

        # convert config to pass config class
        # this will validate the config and convert to the correct types
        config = self._config_class(**config)

        if model_handler.hf_config is None or model_handler.hf_config.model_name is None:
            logger.info("SliceGPT only supports select HuggingFace models")
            return model_handler

        model = model_handler.load_model()
        original_param_count = sum(int(p.nelement()) for p in model.parameters())
        logger.info(f'Original model parameters: {original_param_count:,d}')

        from slicegpt.hf_utils import get_model_adapter
        from slicegpt import layernorm_fusion, rotate
        from slicegpt.slicing_scheduler import ConstSlicingScheduler

        model_adapter = get_model_adapter(model_handler.hf_config.model_name, model)

        # replace and fuse layers
        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)

        # compute new embedding dimension given the desired sparsity level
        sparsity = config.sparsity
        new_embedding_dim = int((1 - sparsity) * model_adapter.hidden_size)
        # round (down) to the nearest multiple of round_interval
        round_interval = config.round_interval
        new_embedding_dim -= new_embedding_dim % round_interval
        logger.info(f"New embedding dimension: {new_embedding_dim} (sparsity {100*(1 - new_embedding_dim / model_adapter.hidden_size):.4f} %)")
        schedular = ConstSlicingScheduler(new_embedding_dim)

        # get datasets
        train_dataset = self.get_datasets(config, data_root)

        # rotate and slice
        rotate.rotate_and_slice(model_adapter, train_dataset, schedular, final_orientation=config.final_orientation)
        sliced_param_count = sum(int(p.nelement()) for p in model.parameters())
        sliced_fraction = 1.0 - sliced_param_count / original_param_count
        logger.info(f'Sliced model parameters: {sliced_param_count:,d} (sliced fraction {sliced_fraction:.4f})')

        # return PyTorchModelHandler
        model.save_pretrained(output_model_path)
        model_config = model.to_json()["config"]
        model_config.model_path = output_model_path
        return PyTorchModelHandler(**model_config, model_file_format=ModelFileFormat.PYTORCH_SLICE_GPT_MODEL)
