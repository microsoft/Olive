# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict

import torch

from olive.common.hf.adapter import ModelAdapter
from olive.common.utils import StrEnumBase, set_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf

logger = logging.getLogger(__name__)

# ruff: noqa: N806


class RotateBase(Pass):
    """Base class for rotation passes."""

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
                type_=RotateBase.RotateMode,
                default_value=RotateBase.RotateMode.HADAMARD,
                description="Rotation method to use. Default value is 'hadamard'.",
            ),
        }

    @torch.no_grad()
    def rotate_model(self, model: HfModelHandler, config: Dict[str, Any]):
        """Create a new model with rotate modules.

        :param model: HfModelHandler: The model to rotate.
        :param config: Dict[str, Any]: The configuration for the rotation.
        :return: ModelAdapter with the rotated model, rotation parameters, and save replacements.
        """
        from olive.passes.pytorch.rotate_utils import RotateEmbed, RotateLinear, fuse_layer_norms, get_orthogonal_matrix

        if model.adapter_path:
            logger.info(
                "Model has adapters but %s does not support adapters. Rotating without adapters. The original"
                " adapters will be used as is with the rotated base model.",
                self.__class__.__name__,
            )

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
        rotation_params = {}
        R1 = torch.nn.Parameter(
            get_orthogonal_matrix(model_adapter.hidden_size, config["rotate_mode"], model_adapter.model.device)
        )
        rotation_params["R1"] = R1

        # rotate embeddings and lm_head
        for embed, embed_name in zip(*model_adapter.get_embeds(return_name=True)):
            # embed_tokens @ R1
            set_attr(model_adapter.model, embed_name, RotateEmbed(embed, R1))
        # R1^-1 @ Whead
        lm_head, lm_head_name = model_adapter.get_lm_head(return_name=True)
        set_attr(model_adapter.model, lm_head_name, RotateLinear(lm_head, Q_pre=R1))

        # need v_proj to be rotated separately, so unpack if necessary
        model_adapter.maybe_unpack_qkv()

        # rotate the hidden layers
        for i, layer_adapter in enumerate(model_adapter.get_layer_adapters()):
            R2 = None
            for linear_idx, (linear, linear_name) in enumerate(
                zip(*layer_adapter.get_attention_inputs(return_name=True))
            ):
                # R1^-1 @ Wq, R1^-1 @ Wk, R1^-1 @ Wv @ R2
                if linear_idx == 2 and getattr(linear, "bias", None) is None:
                    # original implementation ignores bias but output doesn't match both when bias is
                    # rotated headwise and when it is not, so we skip it for now
                    # not really an issue since bias is not present in most models
                    R2 = torch.nn.Parameter(
                        get_orthogonal_matrix(model_adapter.head_dim, config["rotate_mode"], model_adapter.model.device)
                    )
                    rotation_params[f"R2_{i}"] = R2
                set_attr(
                    layer_adapter.layer,
                    linear_name,
                    RotateLinear(linear, Q_pre=R1, Q_post=R2 if linear_idx == 2 else None),
                )

            for linear, linear_name in zip(*layer_adapter.get_attention_outputs(return_name=True)):
                # R2^-1 @ Wo @ R1
                set_attr(layer_adapter.layer, linear_name, RotateLinear(linear, Q_pre=R2, Q_post=R1))

            for linear, linear_name in zip(*layer_adapter.get_mlp_inputs(return_name=True)):
                # R1^-1 @ Wup, R1^-1 @ Wgate
                set_attr(layer_adapter.layer, linear_name, RotateLinear(linear, Q_pre=R1))

            for linear, linear_name in zip(*layer_adapter.get_mlp_outputs(return_name=True)):
                # Wdown @ R1
                set_attr(layer_adapter.layer, linear_name, RotateLinear(linear, Q_post=R1))

        return (
            model_adapter,
            rotation_params,
            [((RotateEmbed, RotateLinear), lambda x: x.create_merged("cuda" if torch.cuda.is_available() else "cpu"))],
        )


class QuaRot(RotateBase):
    """Rotate model using QuaRot.

    See https://arxiv.org/pdf/2404.00456 for more details on the algorithm. Only offline weight rotation is supported.

    This pass only supports HfModelHandler.
    """

    @torch.no_grad()
    def _run_for_config(self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str) -> HfModelHandler:
        model_adapter, _, save_replacements = self.rotate_model(model, config)

        # save the model
        model_adapter.save_model(output_model_path, replacements=save_replacements)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)
