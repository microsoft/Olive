# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from olive.common.utils import find_first_matched_value, get_attr, replace_submodules, set_attr

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# ruff: noqa: RUF012


# TODO(jambayk): consider always returning the name of the submodule
def get_submodules(module: nn.Module, mapping: Dict, key: str, return_name: bool = False, return_name_prefix: str = ""):
    names = mapping.get(key, mapping["default"])

    if isinstance(names, str):
        submodules = get_attr(module, names, fail_on_not_found=True)
        names = f"{return_name_prefix}{names}"
    else:
        submodules = [get_attr(module, name, fail_on_not_found=True) for name in names]
        names = [f"{return_name_prefix}{name}" for name in names]

    return submodules if not return_name else (submodules, names)


class UnpackedQKV(nn.Module):
    def __init__(self, qkv: nn.Module, num_attn_heads: int, num_key_value_heads: int, head_dim: int):
        super().__init__()
        q_size = num_attn_heads * head_dim
        kv_size = num_key_value_heads * head_dim

        def create_proj(start, end):
            proj = nn.Linear(q_size, end - start)
            proj.weight = nn.Parameter(qkv.weight[start:end], requires_grad=qkv.weight.requires_grad)
            proj.bias = (
                None if qkv.bias is None else nn.Parameter(qkv.bias[start:end], requires_grad=qkv.bias.requires_grad)
            )
            return proj

        self.q_proj = create_proj(0, q_size)
        self.k_proj = create_proj(q_size, q_size + kv_size)
        self.v_proj = create_proj(q_size + kv_size, q_size + 2 * kv_size)

    def forward(self, hidden_states):
        return torch.cat([self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)], dim=-1)

    @torch.no_grad()
    def create_packed(self) -> nn.Linear:
        qkv = nn.Linear(
            self.q_proj.in_features + self.k_proj.in_features + self.v_proj.in_features,
            self.q_proj.out_features,
        )
        qkv.weight = nn.Parameter(
            torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0),
            requires_grad=self.q_proj.weight.requires_grad,
        )
        qkv.bias = (
            None
            if self.q_proj.bias is None
            else nn.Parameter(
                torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                requires_grad=self.q_proj.bias.requires_grad,
            )
        )
        return qkv


class LayerAdapter:
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
        if isinstance(attention_inputs[0], UnpackedQKV):
            names = [f"{names[0]}.{part}" for part in ["q_proj", "k_proj", "v_proj"]]
            attention_inputs = [attention_inputs[0].q_proj, attention_inputs[0].k_proj, attention_inputs[0].v_proj]
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
        return get_submodules(
            self.mlp, self.MLP_INPUTS, self.model_type, return_name=return_name, return_name_prefix=f"{self.mlp_name}."
        )

    def get_mlp_outputs(self, return_name: bool = True):
        return get_submodules(
            self.mlp, self.MLP_OUTPUTS, self.model_type, return_name=return_name, return_name_prefix=f"{self.mlp_name}."
        )


class ModelAdapter:

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
    EMBEDDINGS = {
        "default": ["model.embed_tokens"],
        "bloom": ["transformer.word_embeddings", "transformer.word_embeddings_layernorm"],
        "falcon": ["transformer.word_embeddings"],
        "gpt2": ["transformer.wte", "transformer.wpe"],
        "gpt_neox": ["gpt_neox.embed_in"],
        "gptj": ["transformer.wte"],
        "opt": ["model.decoder.embed_tokens", "model.decoder.embed_positions"],
        "qwen": ["transformer.wte", "transformer.rotary_emb"],
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

    def __init__(self, config: Union[PretrainedConfig, Dict]):
        self.config = config if isinstance(config, PretrainedConfig) else PretrainedConfig.from_dict(config)
        self.model_type = find_first_matched_value(self.config, "model_type")

        # model attributes
        self.hidden_size = find_first_matched_value(self.config, self.HIDDEN_SIZE_NAMES)
        self.num_attention_heads = find_first_matched_value(self.config, self.NUM_ATTENTION_HEADS_NAMES)
        self.num_key_value_heads = (
            find_first_matched_value(self.config, self.NUM_KEY_VALUE_HEADS_NAMES) or self.num_attention_heads
        )
        self.head_dim = (
            find_first_matched_value(self.config, self.HEAD_DIM_NAMES) or self.hidden_size // self.num_attention_heads
        )
        self.num_hidden_layers = find_first_matched_value(self.config, self.NUM_HIDDEN_LAYER_NAMES)

        self._model = None
        self._layer_adapters = None

    @property
    def model(self) -> "PreTrainedModel":
        if self._model is None:
            raise ValueError("Model is not set. Please set the model using set_model method.")

        return self._model

    def set_model(self, model: "PreTrainedModel"):
        self._model = model
        self._layer_adapters = [LayerAdapter(layer, self.model_type) for layer in self.get_layers(False)]

    def get_embeds(self, return_name: bool = True):
        return get_submodules(self.model, self.EMBEDDINGS, self.model_type, return_name=return_name)

    def get_lm_head(self, return_name: bool = True):
        return get_submodules(self.model, self.LM_HEAD, self.model_type, return_name=return_name)

    def get_pre_head_layernorm(self, return_name: bool = True):
        return get_submodules(self.model, self.PRE_HEAD_LAYERNORM, self.model_type, return_name=return_name)

    def get_layers(self, return_name: bool = True):
        return get_submodules(self.model, self.LAYERS, self.model_type, return_name=return_name)

    def get_layer_adapters(self):
        if self._layer_adapters is None:
            raise ValueError("Layer adapters are not set. Please set the model using set_model method.")

        return self._layer_adapters

    def maybe_untie_word_embeddings(self):
        if self.config.tie_word_embeddings:
            self.config.tie_word_embeddings = False

            self.get_lm_head(False).weight.data = self.get_embeds(False)[0].weight.data.clone()
            logger.debug("Untied word embeddings.")

    def maybe_unpack_qkv(self):
        for layer_adapter in self.get_layer_adapters():
            attn_inputs, attn_input_names = layer_adapter.get_attention_inputs()

            if len(attn_inputs) != 1 or not isinstance(attn_inputs[0], nn.Linear):
                return

            set_attr(
                layer_adapter.layer,
                attn_input_names[0],
                UnpackedQKV(
                    attn_inputs[0],
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                ),
            )

    def save_model(self, output_model_path: str, replacements: List[Tuple[nn.Module, Callable]] = None):
        replacements = replacements or []
        replacements.append([UnpackedQKV, lambda module: module.create_packed()])

        for submodule_type, replacement_fn in replacements:
            logger.debug("Replacing %s with %s", submodule_type, replacement_fn)
            replace_submodules(self.model, submodule_type, replacement_fn)

        self.model.save_pretrained(output_model_path)
