# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from collections import OrderedDict

import config
import torch


def find_weight_by_subname(v_dict, subname):
    value_list = [value for key, value in v_dict.items() if subname in key]
    if len(value_list) != 1:
        raise ValueError("Found too many or too few matches in state dict")
    return value_list[0]


def convert_falcon_weights():
    n_heads = 71
    n_layers = -1
    max_n_layers = 200
    for layer_idx in range(max_n_layers):
        layer_name = f"transformer.h.{layer_idx}."

        if not any(layer_name in key for key in config.state_dict):
            n_layers = layer_idx
            break

        if layer_idx == max_n_layers - 1:
            raise ValueError(f"Number of layer is larger than {max_n_layers}")

    new_dict = OrderedDict()
    for layer_idx in range(n_layers):
        attn_norm_weight = find_weight_by_subname(config.state_dict, f".h.{layer_idx}.input_layernorm.weight")
        attn_norm_bias = find_weight_by_subname(config.state_dict, f".h.{layer_idx}.input_layernorm.bias")

        kqv_weight = find_weight_by_subname(config.state_dict, f".h.{layer_idx}.self_attention.query_key_value.weight")

        attn_linear_weight = find_weight_by_subname(config.state_dict, f".h.{layer_idx}.self_attention.dense.weight")

        to_4h_weight = find_weight_by_subname(config.state_dict, f".h.{layer_idx}.mlp.dense_h_to_4h.weight")

        to_h_weight = find_weight_by_subname(config.state_dict, f".h.{layer_idx}.mlp.dense_4h_to_h.weight")

        new_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = attn_norm_weight.to(torch.float16)
        new_dict[f"model.layers.{layer_idx}.input_layernorm.bias"] = attn_norm_bias.to(torch.float16)

        hidden_size = kqv_weight.shape[-1]
        kqv_weight = kqv_weight.view(n_heads + 2, -1, hidden_size)
        head_dim = int(hidden_size / n_heads)

        new_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = (
            kqv_weight[:-2, :, :].reshape([hidden_size, hidden_size]).to(torch.float16)
        )
        new_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = (
            kqv_weight[[-2], :, :].reshape([head_dim, hidden_size]).to(torch.float16)
        )
        new_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = (
            kqv_weight[[-1], :, :].reshape([head_dim, hidden_size]).to(torch.float16)
        )

        new_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = attn_linear_weight.to(torch.float16)

        new_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = to_4h_weight.to(torch.float16)

        new_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = to_h_weight.to(torch.float16)

    new_dict["model.norm.weight"] = config.state_dict["transformer.ln_f.weight"].to(torch.float16)

    new_dict["model.norm.bias"] = config.state_dict["transformer.ln_f.bias"].to(torch.float16)

    new_dict["model.embed_tokens.weight"] = config.state_dict["transformer.word_embeddings.weight"].to(torch.float16)

    new_dict["lm_head.weight"] = config.state_dict["lm_head.weight"].to(torch.float16)
    return new_dict
