# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict

import torch

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf

logger = logging.getLogger(__name__)


class MergeAdapterWeights(Pass):
    """Merge adapter weights into the base model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {}

    @torch.no_grad()
    def _run_for_config(self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str) -> HfModelHandler:
        if not model.adapter_path:
            raise RuntimeError(
                "No adapter path found in the model. Please check your input "
                "model type or remove `MergeAdapterWeights` from passes configs"
            )
        pytorch_model = model.load_model()
        merged_model = pytorch_model.merge_and_unload()

        model.save_metadata(output_model_path)
        merged_model.save_pretrained(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=None)
