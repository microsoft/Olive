# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from olive.common.hf.wrapper import ModelWrapper
from olive.common.utils import StrEnumBase, cleanup_memory, set_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf

logger = logging.getLogger(__name__)

# ruff: noqa: N806


class RotateBase(Pass):
    """Base class for rotation passes. Can be followed by a pass such as GPTQ to quantize the rotated model weights."""

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
    def rotate_model(self, model: HfModelHandler, rotate_mode: str, seed: int):
        """Create a new model with rotate modules.

        :param model: HfModelHandler: The model to rotate.
        :param rotate_mode: str: The rotation method to use.
        :param seed: int: The random seed for the rotation.
        :return: ModelWrapper with the rotated model, rotation parameters, and save replacements.
        """
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

        # create model wrapper
        model_wrapper = ModelWrapper.from_model(model.load_model(cache_model=True))

        # fuse layernorms into adjacent linear layers
        self.fuse_layer_norms(model_wrapper)

        # rotate the model
        torch.manual_seed(seed)
        rotation_params = []
        R1 = torch.nn.Parameter(
            self.get_orthogonal_matrix(model_wrapper.hidden_size, rotate_mode, model_wrapper.model.device)
        )
        rotation_params.append(R1)

        # rotate embeddings and lm_head
        for embed, embed_name in zip(*model_wrapper.get_embeds()):
            # embed_tokens @ R1
            set_attr(model_wrapper.model, embed_name, RotateEmbed(embed, R1))
        # R1^-1 @ Whead
        lm_head, lm_head_name = model_wrapper.get_lm_head()
        set_attr(model_wrapper.model, lm_head_name, RotateLinear(lm_head, Q_pre=R1))

        # need v_proj to be rotated separately, so unpack if necessary
        model_wrapper.maybe_unpack_qkv()

        # rotate the hidden layers
        for layer_wrapper in model_wrapper.get_layer_wrappers():
            R2 = None
            for linear_idx, (linear, linear_name) in enumerate(zip(*layer_wrapper.get_attention_inputs())):
                # R1^-1 @ Wq, R1^-1 @ Wk, R1^-1 @ Wv @ R2
                if linear_idx == 2 and getattr(linear, "bias", None) is None:
                    # original implementation ignores bias but output doesn't match both when bias is
                    # rotated headwise and when it is not, so we skip it for now
                    # not really an issue since bias is not present in most models
                    R2 = torch.nn.Parameter(
                        self.get_orthogonal_matrix(model_wrapper.head_dim, rotate_mode, model_wrapper.model.device)
                    )
                    rotation_params.append(R2)
                set_attr(
                    layer_wrapper.layer,
                    linear_name,
                    RotateLinear(linear, Q_pre=R1, Q_post=R2 if linear_idx == 2 else None),
                )

            for linear, linear_name in zip(*layer_wrapper.get_attention_outputs()):
                # R2^-1 @ Wo @ R1
                set_attr(layer_wrapper.layer, linear_name, RotateLinear(linear, Q_pre=R2, Q_post=R1))

            for linear, linear_name in zip(*layer_wrapper.get_mlp_inputs()):
                # R1^-1 @ Wup, R1^-1 @ Wgate
                set_attr(layer_wrapper.layer, linear_name, RotateLinear(linear, Q_pre=R1))

            for linear, linear_name in zip(*layer_wrapper.get_mlp_outputs()):
                # Wdown @ R1
                set_attr(layer_wrapper.layer, linear_name, RotateLinear(linear, Q_post=R1))

        return (
            model_wrapper,
            rotation_params,
            [((RotateEmbed, RotateLinear), lambda x: x.create_merged("cuda" if torch.cuda.is_available() else "cpu"))],
        )

    @classmethod
    def fuse_ln_linear(cls, layernorm: nn.Module, linear_layers: Iterable[nn.Linear]):
        """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
        for linear in linear_layers:
            linear_dtype = linear.weight.dtype

            # Calculating new weight and bias
            W_ = linear.weight.data.double()
            linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

            if hasattr(layernorm, "bias"):
                if linear.bias is None:
                    linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
                linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
                linear.bias.data = linear.bias.data.to(linear_dtype)

        # update layernorm weight and bias
        layernorm.weight.data = torch.ones_like(layernorm.weight.data)
        if hasattr(layernorm, "bias"):
            layernorm.bias = None

    @classmethod
    def fuse_layer_norms(cls, model_wrapper: ModelWrapper):
        """Fuse layernorms into adjacent linear layers."""
        # TODO(jambayk): should we support models with layernorms? these require:
        # - subtracting mean from embedding
        # - baking mean into output layers
        # - replacing layernorm with RMSNorm
        # Model architecture changes are required
        if isinstance(model_wrapper.get_pre_head_layernorm(False), nn.LayerNorm):
            raise ValueError("Model uses LayerNorm. Only RMSNorm fusion is supported.")

        logger.debug("Fusing layernorms into adjacent linear layers")

        # untie embedding and lm head
        model_wrapper.maybe_untie_word_embeddings()

        # Layers: Fuse layernorms into adjacent linear layers
        for layer_wrapper in model_wrapper.get_layer_wrappers():
            cls.fuse_ln_linear(layer_wrapper.get_first_layer_norm(False), layer_wrapper.get_attention_inputs(False))
            cls.fuse_ln_linear(layer_wrapper.get_second_layer_norm(False), layer_wrapper.get_mlp_inputs(False))

        # LM Head: Fuse layernorm into linear layer
        cls.fuse_ln_linear(model_wrapper.get_pre_head_layernorm(False), [model_wrapper.get_lm_head(False)])

    @staticmethod
    def get_orthogonal_matrix(size: int, mode: str, device: torch.device) -> torch.Tensor:
        """Get an orthogonal matrix of the specified size.

        Supported modes:
        - random: generate a random orthogonal matrix
        - hadamard: generate a random Hadamard matrix
        """
        if mode == "random":
            # First, we generate a random matrix with entries from a standard distribution.
            # Then, we use QR decomposition to obtain an orthogonal matrix.
            # Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
            cleanup_memory()
            random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
            q, r = torch.linalg.qr(random_matrix)  # pylint: disable=not-callable
            q *= torch.sign(torch.diag(r)).unsqueeze(0)
            return q
        elif mode == "hadamard":
            from olive.passes.pytorch.hadamard_utils import random_hadamard_matrix

            return random_hadamard_matrix(size, device)
        else:
            raise ValueError(f"Unknown mode {mode}")


class QuaRot(RotateBase):
    """Rotate model using QuaRot.

    See https://arxiv.org/pdf/2404.00456 for more details on the algorithm. Only offline weight rotation is supported.
    Can be followed by a pass such as GPTQ to quantize the rotated model weights.

    This pass only supports HfModelHandler.
    """

    @torch.no_grad()
    def _run_for_config(self, model: HfModelHandler, config: Dict[str, Any], output_model_path: str) -> HfModelHandler:
        model_wrapper, _, save_replacements = self.rotate_model(model, config["rotate_mode"], config["seed"])

        # save the model
        model_wrapper.save_model(output_model_path, replacements=save_replacements)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)


# Rotated Embedding and Linear Layers


def rotate_weight(
    weight: Union[torch.Tensor, nn.Parameter],
    Q: Union[torch.Tensor, nn.Parameter],
    pre: bool = True,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, nn.Parameter]:
    """Rotate the weight matrix by Q.

    :param weight: weight matrix. Shape is (out_features, in_features). Equivalent to W^T in y = xW + b
    :param Q: rotation matrix. If the Q dimension is different from the weight dimension, head-wise rotation
        is performed.
    :pre: whether to apply the rotation before the linear operation.
        True for pre-rotation: y' = (xQ^-1)W + b = x(Q^TW) + b
        False for post-rotation: y' = yQ = x(WQ) + bQ
    :param device: device to use for the computation.
    """
    dtype = weight.dtype
    original_device = weight.device
    out_features, in_features = weight.shape
    q_features = Q.shape[0]

    head_wise = (in_features != q_features) if pre else (out_features != q_features)

    # Convert to double precision, optionally move to device
    to_kwargs = {"dtype": torch.float64}
    if device is not None:
        to_kwargs["device"] = device
    weight = weight.to(**to_kwargs)
    Q = Q.to(**to_kwargs)

    if pre:
        # Q^T @ W = (W^T @ Q)^T = (weight @ Q)^T
        if head_wise:
            weight = weight.reshape(out_features, -1, q_features)
            weight = torch.matmul(weight, Q).reshape(out_features, -1)
        else:
            weight = torch.matmul(weight, Q)
    else:
        # W @ Q = (Q^T @ W^T)^T = (Q^T @ weight)^T
        if head_wise:
            weight = weight.t().reshape(in_features, -1, q_features)
            weight = torch.matmul(weight, Q).reshape(in_features, -1).t()
        else:
            weight = torch.matmul(Q.t(), weight)

    # Convert back to original precision and device
    to_kwargs = {"dtype": dtype}
    if device is not None:
        to_kwargs["device"] = original_device
    return weight.to(**to_kwargs)


class RotateEmbed(nn.Module):
    """Embedding layer with pre-rotation."""

    def __init__(self, embedding: nn.Embedding, Q: nn.Parameter):
        super().__init__()
        self.embedding = embedding
        self.Q = Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.matmul(self.embedding(x).to(torch.float64), self.Q.to(torch.float64))).to(
            self.embedding.weight.dtype
        )

    @torch.no_grad()
    def create_merged(self, device) -> nn.Embedding:
        """Create a merged embedding layer with the rotation matrix."""
        embed = nn.Embedding.from_pretrained(
            rotate_weight(self.embedding.weight.data, self.Q, pre=True, device=device).contiguous(),
        )
        cleanup_memory()
        return embed


class RotateLinear(nn.Module):
    """Linear layer with pre/post rotations."""

    def __init__(self, linear: nn.Linear, Q_pre: Optional[nn.Parameter] = None, Q_post: Optional[nn.Parameter] = None):
        super().__init__()
        self.linear = linear
        self.Q_pre = Q_pre
        self.Q_post = Q_post

    def get_rotated_weights(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = self.linear.weight
        bias = self.linear.bias if hasattr(self.linear, "bias") and self.linear.bias is not None else None

        if self.Q_pre is not None:
            weight = rotate_weight(weight, self.Q_pre, pre=True, device=device)
        if self.Q_post is not None:
            weight = rotate_weight(weight, self.Q_post, pre=False, device=device)
            if bias is not None:
                bias = rotate_weight(bias.unsqueeze(1), self.Q_post, pre=False, device=device).squeeze(1)

        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, *self.get_rotated_weights())  # pylint: disable=not-callable

    @torch.no_grad()
    def create_merged(self, device) -> nn.Linear:
        """Create a merged linear layer with the rotation matrices."""
        weight, bias = self.get_rotated_weights(device)

        linear = nn.Linear(weight.size(1), weight.size(0), bias is not None)
        linear.weight = nn.Parameter(weight.contiguous())
        if bias is not None:
            linear.bias = nn.Parameter(bias.contiguous())

        return linear
