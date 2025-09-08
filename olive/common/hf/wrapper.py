# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import torch
from torch import nn
from transformers import PretrainedConfig

from olive.common.quant.linear import QuantLinear
from olive.common.utils import find_first_matched_value, get_attr, replace_submodules, set_attr

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# ruff: noqa: RUF012


def get_submodules(
    module: nn.Module,
    mapping: dict,
    key: str,
    return_name: bool = False,
    return_name_prefix: str = "",
    fail_on_not_found: bool = True,
):
    names = mapping.get(key, mapping["default"])

    if isinstance(names, str):
        submodules = get_attr(module, names, fail_on_not_found=fail_on_not_found)
        names = f"{return_name_prefix}{names}"
    else:
        submodules = [get_attr(module, name, fail_on_not_found=fail_on_not_found) for name in names]
        names = [f"{return_name_prefix}{name}" for name in names]

    return submodules if not return_name else (submodules, names)


class SplitLinear(nn.Module):
    """Split a single linear layer into multiple linear layers along the output dimension."""

    def __init__(self, linear: nn.Linear | QuantLinear, names: list[str], sizes: list[int]):
        super().__init__()
        assert sum(sizes) == linear.out_features, "Sizes must sum to the output features of the linear layer."
        self.split_projs = nn.ModuleDict()

        if isinstance(linear, nn.Linear):
            weight = linear.weight
        elif isinstance(linear, QuantLinear):
            weight, scale, zero_point = linear.get_unpacked_params(transpose=False)
        else:
            raise ValueError("linear must be an instance of nn.Linear or QuantLinear.")

        start = 0
        for name, size in zip(names, sizes):
            if isinstance(linear, nn.Linear):
                proj = nn.Linear(linear.in_features, size)
                proj.weight = nn.Parameter(weight[start : start + size], requires_grad=linear.weight.requires_grad)
            else:
                scale_zp_slice = slice(start, start + size) if linear.quantizer.group_size == 0 else slice(None)
                proj = QuantLinear.from_tensors(
                    weight[start : start + size],
                    scale[scale_zp_slice],
                    zero_point[scale_zp_slice],
                    bits=linear.quantizer.bits,
                    symmetric=linear.quantizer.symmetric,
                    group_size=linear.quantizer.group_size,
                )
            proj.bias = (
                None
                if linear.bias is None
                else nn.Parameter(linear.bias[start : start + size], requires_grad=linear.bias.requires_grad)
            )
            self.split_projs[name] = proj
            start += size

    def forward(self, x):
        return torch.cat([proj(x) for proj in self.split_projs.values()], dim=-1)

    @torch.no_grad()
    def create_joined(self) -> nn.Linear | QuantLinear:
        """Join the split linear layers back into a single linear layer."""
        all_projs = list(self.split_projs.values())
        if isinstance(all_projs[0], nn.Linear):
            weight = torch.cat([proj.weight for proj in all_projs], dim=0)
            joined = nn.Linear(all_projs[0].in_features, weight.shape[0])
            joined.weight = nn.Parameter(weight, requires_grad=all_projs[0].weight.requires_grad)
        else:
            for key in ["bits", "symmetric", "group_size"]:
                assert all(
                    getattr(proj.quantizer, key) == getattr(all_projs[0].quantizer, key) for proj in all_projs
                ), f"All QuantLinear layers must have the same {key} setting."
            weights = []
            scales = []
            zero_points = []
            for proj in all_projs:
                weight, scale, zero_point = proj.get_unpacked_params(transpose=False)
                weights.append(weight)
                scales.append(scale)
                zero_points.append(zero_point)
            if all_projs[0].quantizer.group_size == 0:
                for scale, zero_point in zip(scales[1:], zero_points[1:]):
                    assert torch.equal(scales[0], scale), (
                        "All QuantLinear layers must have the same scales when group_size is 0."
                    )
                    assert torch.equal(zero_points[0], zero_point), (
                        "All QuantLinear layers must have the same zero_points when group_size is 0."
                    )
                scales = [scales[0]]
                zero_points = [zero_points[0]]
            joined = QuantLinear.from_tensors(
                torch.cat(weights, dim=0),
                torch.cat(scales, dim=0),
                torch.cat(zero_points, dim=0),
                bits=all_projs[0].quantizer.bits,
                symmetric=all_projs[0].quantizer.symmetric,
                group_size=all_projs[0].quantizer.group_size,
            )
        joined.bias = (
            None
            if all_projs[0].bias is None
            else nn.Parameter(
                torch.cat([proj.bias for proj in all_projs], dim=0), requires_grad=all_projs[0].bias.requires_grad
            )
        )
        return joined


class LayerWrapper:
    """Wrapper for transformer layer block."""

    FIRST_LAYER_NORM = {"default": "input_layernorm", "gpt2": "ln_1", "opt": "self_attn_layer_norm", "qwen": "ln_1"}
    SECOND_LAYER_NORM = {
        "default": "post_attention_layernorm",
        "gemma2": "pre_feedforward_layernorm",
        "gpt2": "ln_2",
        "opt": "final_layer_norm",
        "qwen": "ln_2",
    }
    ATTENTION = {"default": "self_attn", "bloom": "self_attention", "gpt2": "attn", "qwen": "attn"}
    ATTENTION_INPUTS = {
        "default": ["q_proj", "k_proj", "v_proj"],
        "bloom": ["query_key_value"],
        "gpt2": ["c_attn"],
        "phi3": ["qkv_proj"],
        "qwen": ["c_attn"],
    }
    ATTENTION_OUTPUTS = {
        "default": ["o_proj"],
        "bloom": ["dense"],
        "gpt2": ["c_proj"],
        "opt": ["out_proj"],
        "qwen": ["c_proj"],
    }
    MLP = {"default": "mlp", "opt": ""}
    MLP_INPUTS = {
        "default": ["gate_proj", "up_proj"],
        "bloom": ["dense_h_to_4h"],
        "gpt2": ["c_fc"],
        "opt": ["fc1"],
        "phi3": ["gate_up_proj"],
        "qwen": ["w1", "w2"],
    }
    MLP_OUTPUTS = {
        "default": ["down_proj"],
        "bloom": ["dense_4h_to_h"],
        "gpt2": ["c_proj"],
        "opt": ["fc2"],
        "qwen": ["c_proj"],
    }

    def __init__(self, layer: nn.Module, model_type: str):
        # TODO(jambayk): use _layer and property to get the layer?
        self.layer = layer
        self.model_type = model_type

        self.attn, self.attn_name = get_submodules(layer, self.ATTENTION, self.model_type, return_name=True)
        self.mlp, self.mlp_name = get_submodules(layer, self.MLP, self.model_type, return_name=True)

    def get_first_layer_norm(self, return_name: bool = True):
        return get_submodules(self.layer, self.FIRST_LAYER_NORM, self.model_type, return_name=return_name)

    def get_second_layer_norm(self, return_name: bool = True):
        return get_submodules(self.layer, self.SECOND_LAYER_NORM, self.model_type, return_name=return_name)

    def get_attention_inputs(self, return_name: bool = True):
        attention_inputs, names = get_submodules(
            self.attn, self.ATTENTION_INPUTS, self.model_type, return_name=True, return_name_prefix=f"{self.attn_name}."
        )
        if isinstance(attention_inputs[0], SplitLinear):
            names = [f"{names[0]}.split_projs.{part}" for part in attention_inputs[0].split_projs]
            attention_inputs = list(attention_inputs[0].split_projs.values())
        return attention_inputs if not return_name else (attention_inputs, names)

    def get_attention_outputs(self, return_name: bool = True):
        return get_submodules(
            self.attn,
            self.ATTENTION_OUTPUTS,
            self.model_type,
            return_name=return_name,
            return_name_prefix=f"{self.attn_name}.",
        )

    def get_mlp_inputs(self, return_name: bool = True):
        mlp_inputs, names = get_submodules(
            self.mlp, self.MLP_INPUTS, self.model_type, return_name=True, return_name_prefix=f"{self.mlp_name}."
        )
        if isinstance(mlp_inputs[0], SplitLinear):
            names = [f"{names[0]}.split_projs.{part}" for part in mlp_inputs[0].split_projs]
            mlp_inputs = list(mlp_inputs[0].split_projs.values())
        return mlp_inputs if not return_name else (mlp_inputs, names)

    def get_mlp_outputs(self, return_name: bool = True):
        return get_submodules(
            self.mlp, self.MLP_OUTPUTS, self.model_type, return_name=return_name, return_name_prefix=f"{self.mlp_name}."
        )


class ModelWrapper:
    """Wrapper for transformer model."""

    INTERMEDIATE_SIZE_NAMES = ("ffn_hidden_size", "intermediate_size")
    HIDDEN_SIZE_NAMES = ("hidden_size", "dim", "d_model", "n_embd")
    NUM_ATTENTION_HEADS_NAMES = (
        "num_attention_heads",
        "num_heads",
        "n_head",
        "n_heads",
        "encoder_attention_heads",
    )
    NUM_KEY_VALUE_HEADS_NAMES = ("num_key_value_heads",)
    HEAD_DIM_NAMES = ("head_dim",)
    NUM_HIDDEN_LAYER_NAMES = ("num_hidden_layers", "num_layers", "n_layer", "n_layers")
    MAX_LENGTH = {
        "default": "max_position_embeddings",
        "gpt2": "n_positions",
        "gptj": "n_positions",
        "qwen": "seq_length",
    }
    EMBEDDINGS = {
        "default": ["model.embed_tokens"],
        "bloom": ["transformer.word_embeddings", "transformer.word_embeddings_layernorm"],
        "falcon": ["transformer.word_embeddings"],
        "gpt2": ["transformer.wte", "transformer.wpe"],
        "gpt_neox": ["gpt_neox.embed_in"],
        "gptj": ["transformer.wte"],
        "opt": ["model.decoder.embed_tokens", "model.decoder.embed_positions"],
        "qwen": ["transformer.wte"],
    }
    # in newer transformers versions, there is one rotary embedding per model
    ROTARY_EMBEDDING = {
        "default": "model.rotary_emb",
        "falcon": "transformer.rotary_emb",
        "gpt_neox": "gpt_neox.rotary_emb",
        "qwen": "transformer.rotary_emb",
    }
    LM_HEAD = {"default": "lm_head"}
    PRE_HEAD_LAYERNORM = {"default": "model.norm", "gpt2": "transformer.ln_f", "qwen": "transformer.ln_f"}
    LAYERS = {
        "default": "model.layers",
        "bloom": "transformer.h",
        "falcon": "transformer.h",
        "gpt2": "transformer.h",
        "gpt_neox": "gpt_neox.layers",
        "gptj": "transformer.h",
        "opt": "model.decoder.layers",
        "qwen": "transformer.h",
    }

    def __init__(self, config: PretrainedConfig | dict):
        self.config = config if isinstance(config, PretrainedConfig) else PretrainedConfig.from_dict(config)
        self.model_type = find_first_matched_value(self.config, "model_type")

        # model attributes
        self.intermediate_size = find_first_matched_value(self.config, self.INTERMEDIATE_SIZE_NAMES)
        self.hidden_size = find_first_matched_value(self.config, self.HIDDEN_SIZE_NAMES)
        self.num_attention_heads = find_first_matched_value(self.config, self.NUM_ATTENTION_HEADS_NAMES)
        self.num_key_value_heads = (
            find_first_matched_value(self.config, self.NUM_KEY_VALUE_HEADS_NAMES) or self.num_attention_heads
        )
        self.head_dim = (
            find_first_matched_value(self.config, self.HEAD_DIM_NAMES) or self.hidden_size // self.num_attention_heads
        )
        self.num_hidden_layers = find_first_matched_value(self.config, self.NUM_HIDDEN_LAYER_NAMES)
        self.max_length = find_first_matched_value(self.config, self.MAX_LENGTH)

        self._model = None
        self._layer_wrappers = None

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            raise ValueError("Model is not set. Please set the model using set_model method.")

        return self._model

    def set_model(self, model: PreTrainedModel):
        self._model = model
        self._layer_wrappers = [LayerWrapper(layer, self.model_type) for layer in self.get_layers(False)]

    def get_embeds(self, return_name: bool = True):
        return get_submodules(self.model, self.EMBEDDINGS, self.model_type, return_name=return_name)

    def get_rotary_embed(self, return_name: bool = True):
        return get_submodules(
            self.model, self.ROTARY_EMBEDDING, self.model_type, return_name=return_name, fail_on_not_found=False
        )

    def get_lm_head(self, return_name: bool = True):
        return get_submodules(self.model, self.LM_HEAD, self.model_type, return_name=return_name)

    def get_pre_head_layernorm(self, return_name: bool = True):
        return get_submodules(self.model, self.PRE_HEAD_LAYERNORM, self.model_type, return_name=return_name)

    def get_layers(self, return_name: bool = True):
        return get_submodules(self.model, self.LAYERS, self.model_type, return_name=return_name)

    def get_layer_wrappers(self):
        if self._layer_wrappers is None:
            raise ValueError("Layer wrappers are not set. Please set the model using set_model method.")

        return self._layer_wrappers

    def maybe_untie_word_embeddings(self):
        """Untie the word embeddings if they are tied."""
        if self.config.tie_word_embeddings:
            self.config.tie_word_embeddings = False
            self.model.config.tie_word_embeddings = False

            self.get_lm_head(False).weight = nn.Parameter(self.get_embeds(False)[0].weight.clone().detach())
            logger.debug("Untied word embeddings.")

    def maybe_split_qkv(self):
        """Split the QKV projection matrix into separate projections for models like phi3."""
        for layer_wrapper in self.get_layer_wrappers():
            attn_inputs, attn_input_names = layer_wrapper.get_attention_inputs()

            if len(attn_inputs) != 1 or not isinstance(attn_inputs[0], nn.Linear):
                return

            q_size = self.num_attention_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim

            set_attr(
                layer_wrapper.layer,
                attn_input_names[0],
                SplitLinear(
                    attn_inputs[0],
                    ["q_proj", "k_proj", "v_proj"],
                    [q_size, kv_size, kv_size],
                ),
            )

    def maybe_split_mlp_inputs(self):
        """Split the MLP input projection matrix into separate projections for models like phi3."""
        for layer_wrapper in self.get_layer_wrappers():
            mlp_inputs, mlp_input_names = layer_wrapper.get_mlp_inputs()

            if len(mlp_inputs) != 1 or not isinstance(mlp_inputs[0], nn.Linear):
                return

            set_attr(
                layer_wrapper.layer,
                mlp_input_names[0],
                SplitLinear(
                    mlp_inputs[0],
                    ["gate_proj", "up_proj"],
                    [self.intermediate_size, self.intermediate_size],
                ),
            )

    def save_model(self, output_model_path: str, replacements: list[tuple[nn.Module, Callable]] = None):
        """Save the model to the output_model_path with the specified replacements.

        :param output_model_path: Path to save the model.
        :param replacements: List of replacements to apply before saving the model. Each replacement is a tuple of
            (submodule_type, replacement_fn). The replacement_fn should take the submodule as input and return the
            replacement module.
        """
        replacements = replacements or []
        # unpack qkv before saving
        replacements.append([SplitLinear, lambda module: module.create_joined()])

        for submodule_type, replacement_fn in replacements:
            logger.debug("Replacing %s with %s", submodule_type, replacement_fn)
            replace_submodules(self.model, submodule_type, replacement_fn)

        self.model.save_pretrained(output_model_path)

    @classmethod
    def from_model(cls, model: PreTrainedModel) -> ModelWrapper:
        model_wrapper = cls(model.config)
        model_wrapper.set_model(model)
        return model_wrapper
