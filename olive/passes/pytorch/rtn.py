# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from olive.passes import Pass
from olive.passes.pytorch.moe_support import check_moe_layout_support
from olive.passes.pytorch.quant_utils import finalize, get_quantizer_config, prepare_model

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler
    from olive.passes.pass_config import BasePassConfig, PassConfigParam


logger = logging.getLogger(__name__)


class Rtn(Pass):
    """Round-to-nearest quantization."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return get_quantizer_config(allow_embeds=True, allow_moe=True)

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        """Run RTN quantization on the model.

        Args:
            model: The HuggingFace model to quantize.
            config: Configuration object containing quantization parameters.
            output_model_path: Path where the quantized model will be saved.

        Returns:
            HfModelHandler for the quantized model.

        """
        wrapper, qcfg, retie_word_embeddings = prepare_model(model, config, allow_quantized=True)
        # Gate on this invocation's own ``moe`` request, not ``qcfg.moe`` -- ``prepare_model``
        # ORs in any pre-existing checkpoint's ``moe`` flag (see ``quant_utils.prepare_model``),
        # so ``qcfg.moe`` can be True even when this run itself passed ``moe=False``.
        if getattr(config, "moe", False):
            experts_modules = [
                experts
                for layer in wrapper.get_layer_wrappers()
                if (experts := layer.get_experts(return_name=False)) is not None
            ]
            if experts_modules:
                check_moe_layout_support(
                    experts_modules,
                    model_type=wrapper.model_type,
                    operation="RTN MoE quantization",
                )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return finalize(model, output_model_path, wrapper, qcfg, device, retie_word_embeddings=retie_word_embeddings)
