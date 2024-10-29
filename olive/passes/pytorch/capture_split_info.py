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

from olive.common.hf.mappings import MODELS_TO_LAYERS_MAPPING
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
                    " supported transformers models, the default value is the transformers layer block name. Refer to"
                    " olive.common.hf.mappings.MODELS_TO_LAYERS_MAPPING for supported models."
                ),
            ),
            "cost_model": PassConfigParam(
                type_=Union[str, Path],
                category=ParamCategory.PATH,
                description="Path to the cost model csv file. One of num_splits or cost_model is required.",
            ),
            # TODO(jambayk): Get this from the accelerator spec?
            "max_memory": PassConfigParam(
                type_=int,
                description=(
                    "Maximum memory in bytes that can be used by the model. Required if cost_model is provided."
                ),
            ),
        }

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})

        if search_point.get("num_splits") is None and search_point.get("cost_model") is None:
            logger.info("One of num_splits or cost_model is required.")
            return False

        if search_point.get("cost_model") is not None and search_point.get("max_memory") is None:
            logger.info("max_memory is required if cost_model is provided.")
            return False

        return True

    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        split_assignments = None
        if config["num_splits"]:
            split_assignments = self.split_using_num_splits(model, config["num_splits"], config["block_to_split"])
        elif config["cost_model"]:
            split_assignments = self.split_using_cost_model(model, config["cost_model"], config["max_memory"])
        else:
            raise ValueError("One of num_splits or cost_model is required.")

        # create a copy of the iput model and add the split assignments as a new attribute
        model.model = None
        output_model = deepcopy(model)
        output_model.model_attributes = model_attributes = output_model.model_attributes or {}
        model_attributes["split_assignments"] = split_assignments

        return output_model

    def split_using_num_splits(
        self, model: Union[HfModelHandler, PyTorchModelHandler], num_splits: int, block_to_split: Optional[str] = None
    ) -> Dict[str, int]:
        # check for None specifically since "" is a valid value
        if block_to_split is None and isinstance(model, HfModelHandler):
            model_type = model.get_hf_model_type()
            block_to_split = MODELS_TO_LAYERS_MAPPING.get(model_type, None)
        if block_to_split is None:
            raise ValueError("block_to_split is not set and could not be inferred. Please set it manually.")

        # we could get the number of layers for hf model from the model attributes
        # but will just load the model to make the logic simple for now
        # consider loading with meta device to avoid loading the weights
        loaded_model = model.load_model(cache_model=False)
        block = get_attr(loaded_model, block_to_split)
        if block is None:
            raise ValueError(f"block_to_split {block_to_split} not found in model.")
        block_members = [child_name for child_name, _ in block.named_children()]

        split_assignments = {}
        for split_idx, split_members in enumerate(np.array_split(block_members, num_splits)):
            for child_name in split_members:
                split_assignments[f"{block_to_split}.{child_name}".lstrip(".")] = split_idx

        return split_assignments

    def split_using_cost_model(
        self, model: Union[HfModelHandler, PyTorchModelHandler], cost_model: Union[str, Path], max_memory: int
    ) -> Dict[str, int]:
        # will only care about the number of bytes for now
        module_to_bytes = {}
        with open(cost_model) as f:
            reader = csv.DictReader(f)
            for row in reader:
                module_to_bytes[row["module"]] = int(row["num_bytes"])

        split_assignments = {}
        split_idx = 0
        split_bytes = 0
        for name, _ in model.load_model(cache_model=False).named_modules():
            if name not in module_to_bytes:
                continue

            num_bytes = module_to_bytes[name]
            if split_bytes + num_bytes > max_memory:
                split_idx += 1
                split_bytes = 0

            split_assignments[name] = split_idx
            split_bytes += num_bytes
        logger.info("Split the model into %d splits based on the cost model.", split_idx + 1)

        return split_assignments
