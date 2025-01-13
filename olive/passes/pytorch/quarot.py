# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict

import torch

from olive.common.hf.adapter import ModelAdapter
from olive.common.utils import StrEnumBase, cleanup_memory
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf

logger = logging.getLogger(__name__)

# ruff: noqa: N806


class QuaRot(Pass):
    """Rotate model using QuaRot.

    See https://arxiv.org/pdf/2404.00456 for more details on the algorithm. Only offline weight rotation is supported.

    This pass only supports HfModelHandler.
    """

    class RotateMode(StrEnumBase):
        HADAMARD = "hadamard"
        RANDOM = "random"

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "seed": PassConfigParam(
                type_=int,
                default_value=0,
                description="Random seed for rotation. Default value is 0.",
            ),
            "rotate_mode": PassConfigParam(
                type_=QuaRot.RotateMode,
                default_value=QuaRot.RotateMode.HADAMARD,
                description="Rotation method to use. Default value is 'hadamard'.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str) -> HfModelHandler:
        from olive.passes.pytorch.rotate_utils import (
            fuse_layer_norms,
            get_orthogonal_matrix,
            rotate_embed,
            rotate_post_linear,
            rotate_pre_linear,
        )

        adapter_path = None
        if model.adapter_path:
            logger.info(
                "Model has adapters but QuaRot does not support adapters. Rotating without adapters. The original"
                " adapters will be used as is with the rotated base model."
            )
            adapter_path = model.adapter_path

            # create a new input model with the adapter path removed
            model.model = None
            model = deepcopy(model)
            model.set_resource("adapter_path", None)

        # create model adapter and load pytorch model
        model_adapter = ModelAdapter(model.get_hf_model_config())
        model_adapter.set_model(model.load_model(cache_model=False))
        model_adapter.model.eval()

        # fuse layernorms into adjacent linear layers
        fuse_layer_norms(model_adapter)

        # rotate the model
        torch.manual_seed(config["seed"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        R1 = get_orthogonal_matrix(model_adapter.hidden_size, config["rotate_mode"], device)

        # rotate embeddings and lm_head
        for embed in model_adapter.get_embeds():
            # We @ R1
            rotate_embed(embed, R1, device)
        # R1^-1 @ Whead
        rotate_pre_linear(model_adapter.get_lm_head(), R1, device)
        cleanup_memory()

        # rotate the hidden layers
        for i, layer_adapter in enumerate(model_adapter.get_layer_adapters()):
            logger.debug("Rotating layer %d/%d", i + 1, model_adapter.num_hidden_layers)

            # rotate the inputs and outputs of the layer blocks
            for linear in layer_adapter.get_attention_inputs() + layer_adapter.get_mlp_inputs():
                # R1^-1 @ Wq, R1^-1 @ Wk, R1^-1 @ Wv, R1^-1 @ Wup, R1^-1 @ Wgate
                rotate_pre_linear(linear, R1, device)

            for linear in layer_adapter.get_attention_outputs() + layer_adapter.get_mlp_outputs():
                # Wo @ R1, Wdown @ R1
                rotate_post_linear(linear, R1, device)

            # need v_proj to be rotated separately, so unpack if necessary
            layer_adapter.maybe_unpack_qkv()

            v_proj = layer_adapter.get_attention_inputs()[2]
            if hasattr(v_proj, "bias") and v_proj.bias is not None:
                # original implementation ignores bias but output doesn't match both when bias is
                # rotated headwise and when it is not, so we skip it for now
                # not really an issue since bias is not present in most models
                continue

            # headwise rotate value projection
            R2 = get_orthogonal_matrix(model_adapter.head_dim, config["rotate_mode"], device)
            # Wv @ R2
            rotate_post_linear(v_proj, R2, device)
            # R2^-1 @ Wo
            rotate_pre_linear(layer_adapter.get_attention_outputs()[0], R2, device)

            # TODO(jambayk): consider adding a save_model method to ModelAdapter which does the repacking and
            # removing other wrappers
            layer_adapter.maybe_pack_qkv()
        cleanup_memory()

        # save the model
        model_adapter.model.save_pretrained(output_model_path)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=adapter_path)
