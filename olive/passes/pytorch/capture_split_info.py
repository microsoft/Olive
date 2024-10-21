# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict, Union

from olive.common.hf.mappings import MODEL_LAYERS_BLOCK_NAME, NUM_HIDDEN_LAYER_NAMES
from olive.common.utils import find_first_matched_value, get_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
import numpy as np

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
            "layers_block_name": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Name of the layers block to split. Supported model types in"
                    " olive.common.hf.mappings.MODELS_TO_LAYERS_MAPPING are auto-filled"
                ),
            ),
        }

    def _run_for_config(
        self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any], output_model_path: str
    ) -> Union[HfModelHandler, PyTorchModelHandler]:
        layers_block_name = config["layers_block_name"]
        if isinstance(model, HfModelHandler):
            model_type = model.get_hf_model_type()
            if layers_block_name is None:
                layers_block_name = MODEL_LAYERS_BLOCK_NAME.get(model_type, None)
        if not layers_block_name:
            raise ValueError("layers_block_name is not set and could not be inferred. Please set it manually.")

        num_layers = None
        if isinstance(model, HfModelHandler):
            # model attributes already has the hf model config loaded
            num_layers = find_first_matched_value(model.model_attributes, NUM_HIDDEN_LAYER_NAMES)
        if not num_layers:
            # load the models and get the number of layers
            loaded_model = model.load_model(cache_model=False)
            layers = get_attr(loaded_model, layers_block_name)

            if layers is None:
                raise ValueError(f"layers_block_name {layers_block_name} not found in model.")

            num_layers = len(layers)

        split_assignments = {}
        for split_idx, split_members in enumerate(np.array_split(range(num_layers), config["num_splits"])):
            for i in split_members:
                split_assignments[f"{layers_block_name}.{i}"] = split_idx

        # create a copy of the iput model and add the split assignments as a new attribute
        model.model = None
        output_model = deepcopy(model)
        output_model.model_attributes = model_attributes = output_model.model_attributes or {}
        model_attributes["split_assignments"] = split_assignments

        return output_model
