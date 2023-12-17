# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc

import config
import torch
from decoder_model import DecoderModel


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


def get_or_create_decoder_model():
    scale_type = "SquareRootHeadDim"

    # Lazily load the decoder model the first time it's requested. This is necessary because both the cache and
    # no_cache models need to share the same instance in order to export their common weights with the same names.
    # Not doing so would result identical weights having different names in both models, which makes merging them
    # very difficult.
    if config.decoder_model is None:
        config.decoder_model = DecoderModel(
            config.num_layers,
            config.vocab_size,
            config.hidden_size,
            config.num_heads,
            scale_type,
            config.normalization_type,
        )
        config.decoder_model.eval()
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
    past_seq_len = 246
    seq_len = 10
    max_seq_len = past_seq_len + seq_len
    head_size = config.hidden_size // config.num_heads

    return {
        "tokens": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "position_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "attn_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int32),
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
    return {
        "tokens_increment": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "position_ids_increment": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "attn_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int32),
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


# -----------------------------------------------------------------------------
# MERGED DECODERS
# -----------------------------------------------------------------------------


def merged_decoders_inputs(model):
    batch_size = 2
    max_seq_len = 256
    head_size = config.hidden_size // config.num_heads
    seq_len = 10
    past_seq_len = 246

    inputs = {
        "tokens": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "position_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "seqlens_k": torch.ones((batch_size,), dtype=torch.int32) * past_seq_len,
        "total_seq_len": torch.ones((1,), dtype=torch.int32) * max_seq_len,
    }

    for layer_idx in range(config.num_layers):
        inputs[f"cache.{layer_idx}.key"] = torch.rand(
            (batch_size, config.num_key_value_heads, max_seq_len, head_size), dtype=torch.float32
        )
        inputs[f"cache.{layer_idx}.value"] = torch.rand(
            (batch_size, config.num_key_value_heads, max_seq_len, head_size), dtype=torch.float32
        )

    inputs["tokens_increment"] = torch.zeros((batch_size, 1), dtype=torch.int64)
    inputs["position_ids_increment"] = torch.zeros((batch_size, 1), dtype=torch.int64)
    inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)

    return inputs


def merged_decoders_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(merged_decoders_inputs, batch_size)
