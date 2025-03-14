# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import csv
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, Type, Union

import numpy as np

from olive.common.hf.wrapper import ModelWrapper
from olive.common.utils import get_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, ParamCategory, PassConfigParam

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


class CaptureSplitInfo(Pass):
    """Capture the split information of the model layers. Only splits the transformer layers."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "num_splits": PassConfigParam(
                type_=int,
                description="Number of splits to divide the model layers into.",
            ),
            "block_to_split": PassConfigParam(
                type_=Union[str, list[str]],
                default_value=None,
                description=(
                    "Names of the model blocks to split. Children of the block will be divided into the splits. For"
                    " supported transformers models, the default value is the transformers layer block name."
                ),
            ),
            "cost_model": PassConfigParam(
                type_=Union[str, Path],
                category=ParamCategory.PATH,
                description=(
                    "Path to the cost model csv file. One of num_splits or cost_model is required. Must be a csv with"
                    " headers `module,num_params,num_bytes,num_flops` where each row corresponds to the name or a"
                    " module (with no children), the number of parameters, the number of bytes, and the number of"
                    " FLOPs(batch_size=1, seqlen=1) the module uses when in the desired precision."
                ),
            ),
            "unique_embeds_lm_head_splits": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Assign embeddings and lm_head layers to their own splits.",
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: Type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.num_splits is None and config.cost_model is None:
            logger.info("One of num_splits or cost_model is required.")
            return False

        if config.cost_model is not None and accelerator_spec.memory is None:
            logger.info("Accelerator memory is required if cost_model is provided.")
            return False

        return True

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Type[BasePassConfig], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        split_assignments = None
        if config.num_splits:
            split_assignments = self.split_using_num_splits(model, config)
        elif config.cost_model:
            split_assignments = self.split_using_cost_model(model, config)
        else:
            raise ValueError("One of num_splits or cost_model is required.")

        # create a copy of the iput model and add the split assignments as a new attribute
        model.model = None
        output_model = deepcopy(model)
        output_model.model_attributes = model_attributes = output_model.model_attributes or {}
        model_attributes["split_assignments"] = split_assignments

        return output_model

    def split_using_num_splits(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Type[BasePassConfig]
    ) -> Dict[str, int]:
        # consider loading with meta device to avoid loading the weights
        loaded_model = model.load_model(cache_model=False)

        block_to_split = config.block_to_split
        # check for None specifically since "" is a valid value
        if block_to_split is None and isinstance(model, HfModelHandler):
            model_wrapper = ModelWrapper.from_model(loaded_model)
            blocks = [model_wrapper.get_layers()]
        elif block_to_split is None:
            raise ValueError("block_to_split is not set and could not be inferred. Please set it manually.")
        else:
            block_to_splits = block_to_split if isinstance(block_to_split, list) else [block_to_split]
            blocks = [
                (get_attr(loaded_model, block_to_split, fail_on_not_found=True), block_to_split)
                for block_to_split in block_to_splits
            ]

        block_members = [
            f"{block_to_split}.{child_name}".lstrip(".")
            for block, block_to_split in blocks
            for child_name, _ in block.named_children()
        ]

        split_assignments, used_splits, modules_to_exclude = self._init_split_assignments(
            model, loaded_model, config.unique_embeds_lm_head_splits
        )

        for split_idx, split_members in enumerate(np.array_split(block_members, config.num_splits)):
            for member_name in split_members:
                if member_name in modules_to_exclude:
                    continue
                split_assignments[member_name] = split_idx + used_splits

        if config.unique_embeds_lm_head_splits:
            # assign lm_head layer to its own split
            split_assignments["lm_head"] = config.num_splits + used_splits

        return split_assignments

    def split_using_cost_model(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Type[BasePassConfig]
    ) -> Dict[str, int]:
        if self.accelerator_spec.memory is None:
            raise ValueError("Accelerator memory is required to split using cost model.")

        # will only care about the number of bytes for now
        module_to_cost = {}
        with open(config.cost_model) as f:
            reader = csv.DictReader(f)
            for row in reader:
                module_to_cost[row["module"]] = (int(row["num_params"]), int(row["num_bytes"]), int(row["num_flops"]))

        loaded_model = model.load_model(cache_model=False)

        split_assignments, split_idx, modules_to_exclude = self._init_split_assignments(
            model, loaded_model, config.unique_embeds_lm_head_splits
        )

        split_bytes = 0
        node_idx = 0
        mem_intensive = False
        for name, _ in loaded_model.named_modules():
            if name not in module_to_cost or name in modules_to_exclude or name.lower().endswith("dropout"):
                continue

            num_params, num_bytes, num_flops = module_to_cost[name]

            # TODO(jambayk): add num_bytes threshold
            # maybe also a threshold for num_params/num_flops ratio
            curr_mem_intensive = num_params > num_flops > 0

            # change split if:
            #  switching from memory intensive to compute intensive or vice versa
            #  size of split i exceeds the accelerator memory
            # first node is always in split 0
            if node_idx != 0 and (
                (mem_intensive ^ curr_mem_intensive) or (split_bytes + num_bytes > self.accelerator_spec.memory)
            ):
                split_idx += 1
                split_bytes = 0

            split_assignments[name] = split_idx
            split_bytes += num_bytes
            node_idx += 1
            mem_intensive = curr_mem_intensive

        if config.unique_embeds_lm_head_splits:
            # assign lm_head layer to its own split
            split_idx += 1
            split_assignments["lm_head"] = split_idx

        logger.info("Split the model into %d splits based on the cost model.", split_idx + 1)

        return split_assignments

    def _init_split_assignments(
        self,
        model: Union[HfModelHandler, PyTorchModelHandler],
        pytorch_model: "nn.Module",
        unique_embeds_lm_head_splits: bool,
    ) -> Tuple[Dict[str, int], int, set]:
        """Initialize the split assignments for the model.

        :param model: The input model to split.
        :param pytorch_model: The loaded input model.
        :param unique_embeds_lm_head_splits: Whether to assign embedding and lm_head layers to their own splits.
        :return: A dictionary of split assignments, next split index, and set of modules to exclude.
        """
        split_assignments = {}
        split_idx = 0
        modules_to_exclude = set()
        if unique_embeds_lm_head_splits:
            # assign embedding layer to its own split
            if isinstance(model, HfModelHandler):
                embed_names = ModelWrapper.from_model(pytorch_model).get_embeds()[1]
            else:
                embed_names = ["model.embed_tokens"]
            for embed_name in embed_names:
                split_assignments[embed_name] = split_idx
            split_idx += 1

            # exclude embedding and lm head from split calculation
            modules_to_exclude.update([*embed_names, "lm_head"])

        return split_assignments, split_idx, modules_to_exclude
