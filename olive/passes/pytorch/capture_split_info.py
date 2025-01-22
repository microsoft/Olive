# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import csv
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from olive.common.hf.wrapper import ModelWrapper
from olive.common.utils import get_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import ParamCategory, PassConfigParam

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
                type_=str,
                default_value=None,
                description=(
                    "Name of the model block to split. Children of the block will be divided into the splits. For"
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
            "exclude_embeds": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Exclude the embeddings layer/s from the split calculation. Only used with cost_model.",
            ),
            "exclude_lm_head": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Exclude the language model head layer/s from the split calculation. Only used with cost_model."
                ),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: Dict[str, Any],
        accelerator_spec: AcceleratorSpec,
        disable_search: Optional[bool] = False,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec, disable_search):
            return False

        config_cls, _ = cls.get_config_class(accelerator_spec, disable_search)
        config = config_cls(**config)

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
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        split_assignments = None
        if config["num_splits"]:
            split_assignments = self.split_using_num_splits(model, config)
        elif config["cost_model"]:
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
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any]
    ) -> Dict[str, int]:
        # consider loading with meta device to avoid loading the weights
        loaded_model = model.load_model(cache_model=False)

        block_to_split = config["block_to_split"]
        # check for None specifically since "" is a valid value
        if block_to_split is None and isinstance(model, HfModelHandler):
            model_wrapper = ModelWrapper.from_model(loaded_model)
            block, block_to_split = model_wrapper.get_layers()
        elif block_to_split is None:
            raise ValueError("block_to_split is not set and could not be inferred. Please set it manually.")
        else:
            block = get_attr(loaded_model, block_to_split, fail_on_not_found=True)

        block_members = [child_name for child_name, _ in block.named_children()]

        split_assignments = {}
        for split_idx, split_members in enumerate(np.array_split(block_members, config["num_splits"])):
            for child_name in split_members:
                split_assignments[f"{block_to_split}.{child_name}".lstrip(".")] = split_idx

        return split_assignments

    def split_using_cost_model(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any]
    ) -> Dict[str, int]:
        if self.accelerator_spec.memory is None:
            raise ValueError("Accelerator memory is required to split using cost model.")

        # will only care about the number of bytes for now
        module_to_cost = {}
        with open(config["cost_model"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                module_to_cost[row["module"]] = (int(row["num_params"]), int(row["num_bytes"]), int(row["num_flops"]))

        loaded_model = model.load_model(cache_model=False)

        modules_to_exclude = set()
        if config["exclude_embeds"] and isinstance(model, HfModelHandler):
            model_wrapper = ModelWrapper.from_model(loaded_model)
            modules_to_exclude.update(model_wrapper.get_embeds()[1])
        elif config["exclude_embeds"]:
            modules_to_exclude.add("model.embed_tokens")
        if config["exclude_lm_head"]:
            modules_to_exclude.add("lm_head")

        split_assignments = {}
        split_idx = 0
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
        logger.info("Split the model into %d splits based on the cost model.", split_idx + 1)

        return split_assignments
