# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path

import config
import torch
from argmax_sampling_model import ArgmaxSampling
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
# ARGMAX SAMPLING
# -----------------------------------------------------------------------------


def load_argmax_sampling_model(model_path):
    model = ArgmaxSampling()
    model.eval()
    return model


def argmax_sampling_inputs(model):
    batch_size = 2
    vocab_size = 32000
    return torch.zeros((batch_size, vocab_size), dtype=torch.float16)


def argmax_sampling_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(argmax_sampling_inputs, batch_size)


# -----------------------------------------------------------------------------
# DECODER
# -----------------------------------------------------------------------------


def get_or_create_decoder_model():
    num_heads = 32
    vocab_size = 32000
    hidden_size = 4096
    scale_type = "SquareRootHeadDim"

    # Lazily load the decoder model the first time it's requested. This is necessary because both the cache and
    # no_cache models need to share the same instance in order to export their common weights with the same names.
    # Not doing so would result identical weights having different names in both models, which makes merging them
    # very difficult.
    if config.decoder_model is None:
        config.decoder_model = DecoderModel(
            config.num_layers,
            vocab_size,
            hidden_size,
            num_heads,
            scale_type,
            config.normalization_type,
        )
        config.decoder_model.eval()

        script_dir = Path(__file__).resolve().parent
        weights_path = script_dir / "raw_model_data" / config.model_type / f"llama-2-{config.model_type}.pth"

        # We don't use rope.freqs
        state_dict = torch.load(weights_path)

        # Permutation for sliced rotary
        def permute(weight):
            return (
                weight.view(num_heads, hidden_size // num_heads // 2, 2, hidden_size)
                .transpose(1, 2)
                .reshape(hidden_size, hidden_size)
            )

        for layer_idx in range(config.num_layers):
            state_dict[f"layers.{layer_idx}.attention.wq.weight"] = permute(
                state_dict[f"layers.{layer_idx}.attention.wq.weight"]
            )
            state_dict[f"layers.{layer_idx}.attention.wk.weight"] = permute(
                state_dict[f"layers.{layer_idx}.attention.wk.weight"]
            )

        del state_dict["rope.freqs"]
        strict = config.num_layers == 32
        config.decoder_model.load_state_dict(state_dict, strict=strict)

    return config.decoder_model


def load_decoder_model(model_path):
    model = get_or_create_decoder_model()
    model.set_use_cache(False)
    return model


def decoder_inputs(model):
    batch_size = 2
    seq_len = 10
    hidden_size = 4096
    max_seq_len = 256
    num_heads = 32
    head_size = hidden_size // num_heads

    return {
        "tokens": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "position_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "attn_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int32),
        "past_key_values": [
            {
                "key": torch.rand((batch_size, num_heads, max_seq_len, head_size), dtype=torch.float32),
                "value": torch.rand((batch_size, num_heads, max_seq_len, head_size), dtype=torch.float32),
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
    hidden_size = 4096
    max_seq_len = 256
    num_heads = 32
    head_size = hidden_size // num_heads
    return {
        "tokens_increment": torch.zeros((batch_size, 1), dtype=torch.int64),
        "position_ids_increment": torch.zeros((batch_size, 1), dtype=torch.int64),
        "attn_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int32),
        "past_key_values": [
            {
                "key": torch.rand((batch_size, num_heads, max_seq_len, head_size), dtype=torch.float32),
                "value": torch.rand((batch_size, num_heads, max_seq_len, head_size), dtype=torch.float32),
            }
            for _ in range(config.num_layers)
        ],
    }


# -----------------------------------------------------------------------------
# MERGED DECODERS
# -----------------------------------------------------------------------------


def merged_decoders_inputs(model):
    batch_size = 2
    hidden_size = 4096
    max_seq_len = 256
    num_heads = 32
    head_size = hidden_size // num_heads
    seq_len = 10

    inputs = {
        "tokens": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "position_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
        "attn_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int32),
    }

    for layer_idx in range(config.num_layers):
        inputs[f"past_key_values.{layer_idx}.key"] = torch.rand(
            (batch_size, num_heads, max_seq_len, head_size), dtype=torch.float32
        )
        inputs[f"past_key_values.{layer_idx}.value"] = torch.rand(
            (batch_size, num_heads, max_seq_len, head_size), dtype=torch.float32
        )

    inputs["tokens_increment"] = torch.zeros((batch_size, 1), dtype=torch.int64)
    inputs["position_ids_increment"] = torch.zeros((batch_size, 1), dtype=torch.int64)
    inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)

    return inputs


def merged_decoders_data_loader(data_dir, batch_size, *args, **kwargs):
    return RandomDataLoader(merged_decoders_inputs, batch_size)
