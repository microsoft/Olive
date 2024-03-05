# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from collections import OrderedDict

import config

PHI_MAPPING = {
    "transformer.embd.wte.weight": "model.embed_tokens.weight",
    "lm_head.linear": "lm_head",
    "final_layernorm": "norm",
    "transformer": "model",
    ".h.": ".layers.",
    "ln": "input_layernorm",
    "mixer": "self_attn",
    "Wqkv": "query_key_value",
    "dense": "o_proj",
    "mlp.fc1": "mlp.gate_proj",
    "mlp.fc2": "mlp.down_proj",
}


def map_key(origin_key):
    for k, v in PHI_MAPPING.items():
        if k in origin_key:
            origin_key = origin_key.replace(k, v)
    return origin_key


def find_weight_by_subname(v_dict, subname):
    value_list = [value for key, value in v_dict.items() if subname in key]
    if len(value_list) != 1:
        raise ValueError("Found too many or too few matches in state dict")
    return value_list[0]


def convert_phi_weights():
    new_dict = OrderedDict()
    original_weights_keys = sorted(config.state_dict.keys())

    for original_weights_key in original_weights_keys:
        new_key = map_key(original_weights_key)
        new_dict[new_key] = config.state_dict[original_weights_key]

    return new_dict
