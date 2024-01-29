# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import OrderedDict
import gc

import config
import torch
from decoder_model import DecoderModel
from llava_model import LlavaModel
from transformers import AutoConfig


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size), label


# -----------------------------------------------------------------------------
# DECODER
# -----------------------------------------------------------------------------

def find_weight_by_subname(v_dict, subname):
    """
    Find weight in dictionary by subname.

    Arguments:
        v_dict -- dictionary to search.
        subname -- subname to search for.

    Raises:
        ValueError: subname not found, or duplicates.

    Returns:
        The weights for an element with the given subname.
    """
    value_list = [value for key, value in v_dict.items() if subname in key]
    if len(value_list) != 1:
        raise ValueError("Found too many or too few matches in state dict")

    return value_list[0]

def get_or_create_decoder_model():
    scale_type = "SquareRootHeadDim"

    # Lazily load the decoder model the first time it's requested. This is necessary because both the cache and
    # no_cache models need to share the same instance in order to export their common weights with the same names.
    # Not doing so would result identical weights having different names in both models, which makes merging them
    # very difficult.
    if config.decoder_model is None:
        if config.model_type == "llava":
            llava_config = AutoConfig.from_pretrained("llava-hf/llava-1.5-7b-hf")
            config.decoder_model = LlavaModel(llava_config)
        else:
            config.decoder_model = DecoderModel(
                config.model_type,
                config.num_layers,
                config.vocab_size,
                config.hidden_size,
                config.intermediate_size,
                config.num_heads,
                config.num_key_value_heads,
                scale_type,
                config.normalization_type,
                config.epsilon,
                config.apply_residual_connection_post_layernorm,
            )
        config.decoder_model.eval()

        if config.model_type == "falcon":
            n_heads = 71
            n_layers = -1
            max_n_layers = 200
            for layer_idx in range(max_n_layers):
                layer_name = f"transformer.h.{layer_idx}."

                if not any(layer_name in key for key in config.state_dict.keys()):
                    n_layers = layer_idx
                    break

                if layer_idx == max_n_layers - 1:
                    raise ValueError(f"Number of layer is larger than {max_n_layers}")

            print(f"Number of layers in checkpoint {layer_idx}")

            new_dict = OrderedDict()
            for layer_idx in range(n_layers):
                print(f"Processing layer {layer_idx}")

                attn_norm_weight = find_weight_by_subname(
                    config.state_dict, f".h.{layer_idx}.input_layernorm.weight"
                )
                attn_norm_bias = find_weight_by_subname(
                    config.state_dict, f".h.{layer_idx}.input_layernorm.bias"
                )

                kqv_weight = find_weight_by_subname(
                    config.state_dict, f".h.{layer_idx}.self_attention.query_key_value.weight"
                )

                attn_linear_weight = find_weight_by_subname(
                    config.state_dict, f".h.{layer_idx}.self_attention.dense.weight"
                )

                to_4h_weight = find_weight_by_subname(
                    config.state_dict, f".h.{layer_idx}.mlp.dense_h_to_4h.weight"
                )

                to_h_weight = find_weight_by_subname(
                    config.state_dict, f".h.{layer_idx}.mlp.dense_4h_to_h.weight"
                )

                # ------------------------------------------------------

                new_dict[
                    f"model.layers.{layer_idx}.input_layernorm.weight"
                ] = attn_norm_weight.to(torch.float16)
                new_dict[
                    f"model.layers.{layer_idx}.input_layernorm.bias"
                ] = attn_norm_bias.to(torch.float16)

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

                new_dict[
                    f"model.layers.{layer_idx}.self_attn.o_proj.weight"
                ] = attn_linear_weight.to(torch.float16)

                new_dict[
                    f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                ] = to_4h_weight.to(torch.float16)

                new_dict[
                    f"model.layers.{layer_idx}.mlp.down_proj.weight"
                ] = to_h_weight.to(torch.float16)

            new_dict["model.norm.weight"] = config.state_dict[
                "transformer.ln_f.weight"
            ].to(torch.float16)

            new_dict["model.norm.bias"] = config.state_dict[
                "transformer.ln_f.bias"
            ].to(torch.float16)

            new_dict["model.embed_tokens.weight"] = config.state_dict[
                "transformer.word_embeddings.weight"
            ].to(torch.float16)

            new_dict["lm_head.weight"] = config.state_dict[
                "lm_head.weight"
            ].to(torch.float16)

            torch.save(new_dict, "weights.pth")
            config.decoder_model.load_state_dict(new_dict, strict=config.strict_weights_loading)
        else:
            config.decoder_model.load_state_dict(config.state_dict, strict=config.strict_weights_loading)
            
        decoder_model = config.decoder_model

        # Release the memory since we don't need it anymore
        del config.state_dict
    else:
        decoder_model = config.decoder_model

        # Release the memory since we don't need it anymore
        del config.decoder_model

    gc.collect()

    return decoder_model


def load_decoder_model(model_path):
    model = get_or_create_decoder_model()
    model.set_use_cache(False)
    return model


def decoder_inputs(model):
    batch_size = 2
    past_seq_len = 0
    seq_len = 10
    max_seq_len = past_seq_len + seq_len
    head_size = config.hidden_size // config.num_heads

    inputs = {
        "attention_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int64),
        "cache": [
            {
                "key": torch.rand(
                    (batch_size, config.num_key_value_heads, past_seq_len, head_size), dtype=torch.float32
                ),
                "value": torch.rand(
                    (batch_size, config.num_key_value_heads, past_seq_len, head_size), dtype=torch.float32
                ),
            }
            for _ in range(config.num_layers)
        ],
    }

    if config.model_type == "llava":
        channel_count = 3
        image_size = 336
        inputs["pixel_values"] = torch.zeros((batch_size, channel_count, image_size, image_size), dtype=torch.float32)

        # 32000 is the value for the image token and needs to be be there for the model to be successfully generated
        inputs["tokens"] = torch.nn.functional.pad(
            torch.zeros((batch_size, seq_len - 1), dtype=torch.int64), (0, 1), value=32000
        )
    else:
        inputs["position_ids"] = torch.zeros((batch_size, seq_len), dtype=torch.int64)
        inputs["tokens"] = torch.zeros((batch_size, seq_len), dtype=torch.int64)

    return inputs


# -----------------------------------------------------------------------------
# DECODER WITH PAST
# -----------------------------------------------------------------------------


def load_decoder_with_past_model(model_path):
    model = get_or_create_decoder_model()
    model.set_use_cache(True)
    return model


def decoder_with_past_inputs(model):
    batch_size = 2
    past_seq_len = 255
    seq_len = 1
    max_seq_len = past_seq_len + seq_len
    head_size = config.hidden_size // config.num_heads

    inputs = {
        "tokens_increment": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "attention_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int64),
        "cache": [
            {
                "key": torch.rand(
                    (batch_size, config.num_key_value_heads, past_seq_len, head_size), dtype=torch.float32
                ),
                "value": torch.rand(
                    (batch_size, config.num_key_value_heads, past_seq_len, head_size), dtype=torch.float32
                ),
            }
            for _ in range(config.num_layers)
        ],
    }

    if config.model_type != "llava":
        inputs["position_ids_increment"] = torch.zeros((batch_size, seq_len), dtype=torch.int64)

    return inputs


# -----------------------------------------------------------------------------
# MERGED DECODERS
# -----------------------------------------------------------------------------


def merged_decoders_inputs(model):
    batch_size = 2
    head_size = config.hidden_size // config.num_heads
    past_seq_len = 246
    seq_len = 10
    max_seq_len = past_seq_len + seq_len

    inputs = {
        "tokens": torch.zeros((batch_size, seq_len), dtype=torch.int64),
    }

    if config.model_type == "llava":
        channel_count = 3
        image_size = 336

        inputs["attention_mask"] = torch.zeros((batch_size, max_seq_len), dtype=torch.int64)
        inputs["pixel_values"] = torch.zeros((batch_size, channel_count, image_size, image_size), dtype=torch.float32)
    else:
        inputs["position_ids"] = torch.zeros((batch_size, seq_len), dtype=torch.int64)
        inputs["seqlens_k"] = torch.ones((batch_size,), dtype=torch.int32) * past_seq_len
        inputs["total_seq_len"] = torch.ones((1,), dtype=torch.int32) * max_seq_len
        inputs["position_ids_increment"] = torch.zeros((batch_size, 1), dtype=torch.int64)

    for layer_idx in range(config.num_layers):
        inputs[f"cache.{layer_idx}.key"] = torch.rand(
            (batch_size, config.num_key_value_heads, max_seq_len, head_size), dtype=torch.float32
        )
        inputs[f"cache.{layer_idx}.value"] = torch.rand(
            (batch_size, config.num_key_value_heads, max_seq_len, head_size), dtype=torch.float32
        )

    inputs["tokens_increment"] = torch.zeros((batch_size, 1), dtype=torch.int64)
    inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)

    return inputs


def merged_decoders_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(merged_decoders_inputs, batch_size)
