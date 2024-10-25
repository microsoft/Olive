# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict, Union

import numpy as np

from olive.common.hf.mappings import MODELS_TO_LAYERS_MAPPING
from olive.common.utils import get_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class CaptureSplitInfo(Pass):
    """Capture the split information of the model layers. Only splits the transformer layers."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "num_splits": PassConfigParam(
                type_=int,
                required=True,
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
        }

    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        block_to_split = config["block_to_split"]
        # check for None specifically since "" is a valid value
        if block_to_split is None and isinstance(model, HfModelHandler):
            model_type = model.get_hf_model_type()
            block_to_split = MODELS_TO_LAYERS_MAPPING.get(model_type, None)
        if block_to_split is None:
            raise ValueError("block_to_split is not set and could not be inferred. Please set it manually.")

        block_members = []
        # we could get the number of layers for hf model from the model attributes
        # but will just load the model to make the logic simple for now
        # consider loading with meta device to avoid loading the weights
        loaded_model = model.load_model(cache_model=False)
        block = get_attr(loaded_model, block_to_split)
        if block is None:
            raise ValueError(f"block_to_split {block_to_split} not found in model.")
        for child_name, _ in block.named_children():
            block_members.append(child_name)

        split_assignments = {}
        for split_idx, split_members in enumerate(np.array_split(block_members, config["num_splits"])):
            for child_name in split_members:
                split_assignments[f"{block_to_split}.{child_name}".lstrip(".")] = split_idx

        # create a copy of the iput model and add the split assignments as a new attribute
        model.model = None
        output_model = deepcopy(model)
        output_model.model_attributes = model_attributes = output_model.model_attributes or {}
        model_attributes["split_assignments"] = split_assignments

        return output_model
