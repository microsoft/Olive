# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Dict, Type

import torch

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf

logger = logging.getLogger(__name__)


class MergeAdapterWeights(Pass):
    """Merge adapter weights into the base model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {}

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        if not model.adapter_path:
            raise RuntimeError(
                "No adapter path found in the model. Please check your input "
                "model type or remove `MergeAdapterWeights` from passes configs"
            )

        new_load_kwargs = deepcopy(model.load_kwargs.dict()) if model.load_kwargs else {}
        if (
            new_load_kwargs.get("quantization_method") == "bitsandbytes"
            and new_load_kwargs["quantization_config"]["load_in_4bit"]
        ):
            logger.debug(
                "Merging adapter weights for Bitsandbytes 4bit quantized model is not supported. The quantization"
                " config is removed from the load kwargs."
            )
            new_load_kwargs["quantization_method"] = None
            new_load_kwargs["quantization_config"] = None

        pytorch_model = HfModelHandler(
            model_path=model.model_path, task=model.task, adapter_path=model.adapter_path, load_kwargs=new_load_kwargs
        ).load_model()
        merged_model = pytorch_model.merge_and_unload()

        merged_model.save_pretrained(output_model_path)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=None)
