# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import random

import config
import numpy as np
import torch
from datasets import load_dataset
from decoder_model import DecoderModel
from falcon import convert_falcon_weights
from llava_model import LlavaModel
from phi import convert_phi_weights
from phi3 import convert_phi3_weights
from transformers import AutoConfig, AutoTokenizer

from olive.data.registry import Registry


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        label = None
        return self.create_input_func(self.batch_size), label


# -----------------------------------------------------------------------------
# DECODER
# -----------------------------------------------------------------------------


def get_or_create_decoder_model():
    # Lazily load the decoder model the first time it's requested. This is necessary because both the cache and
    # no_cache models need to share the same instance in order to export their common weights with the same names.
    # Not doing so would result identical weights having different names in both models, which makes merging them
    # very difficult.
    if config.decoder_model is None:
        if config.model_type == "falcon":
            config.state_dict = convert_falcon_weights()
        elif config.model_type == "phi":
            config.state_dict = convert_phi_weights()
        elif config.model_type == "phi3":
            config.state_dict = convert_phi3_weights()

        config.has_up_proj = ("model.layers.0.mlp.up_proj.weight" in config.state_dict) or (
            "language_model.model.layers.0.mlp.up_proj.weight" in config.state_dict
        )
        config.has_input_layernorm_bias = "model.layers.0.input_layernorm.bias" in config.state_dict
        config.has_norm_bias = "model.norm.bias" in config.state_dict
        config.has_lm_head_bias = ("language_model.lm_head.bias" in config.state_dict) or (
            "lm_head.bias" in config.state_dict
        )

        if config.model_type == "llava":
            llava_config = AutoConfig.from_pretrained(config.model_id)
            config.decoder_model = LlavaModel(llava_config)
        else:
            config.decoder_model = DecoderModel()
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
    return get_or_create_decoder_model()


def decoder_torch_inputs(model):
    batch_size = 2
    past_seq_len = 0
    sequence_length = 10
    max_seq_len = past_seq_len + sequence_length

    inputs = {
        "attention_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int64),
        "past_key_values": [
            {
                "key": torch.rand(
                    (batch_size, config.num_key_value_heads, past_seq_len, config.head_dim), dtype=torch.float32
                ),
                "value": torch.rand(
                    (batch_size, config.num_key_value_heads, past_seq_len, config.head_dim), dtype=torch.float32
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
        inputs["input_ids"] = torch.nn.functional.pad(
            torch.zeros((batch_size, sequence_length - 1), dtype=torch.int64), (0, 1), value=32000
        )
    else:
        inputs["position_ids"] = torch.zeros((batch_size, sequence_length), dtype=torch.int64)
        inputs["input_ids"] = torch.zeros((batch_size, sequence_length), dtype=torch.int64)

    return inputs


def decoder_ort_inputs(batch_size):
    sequence_length = 10
    max_seq_len = 1024

    inputs = {
        "attention_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int64),
    }

    for layer_idx in range(config.num_layers):
        inputs[f"past_key_values.{layer_idx}.key"] = torch.rand(
            (batch_size, config.num_key_value_heads, max_seq_len, config.head_dim), dtype=torch.float32
        )
        inputs[f"past_key_values.{layer_idx}.value"] = torch.rand(
            (batch_size, config.num_key_value_heads, max_seq_len, config.head_dim), dtype=torch.float32
        )

    if config.model_type == "llava":
        channel_count = 3
        image_size = 336
        inputs["pixel_values"] = torch.zeros((batch_size, channel_count, image_size, image_size), dtype=torch.float32)

        # 32000 is the value for the image token and needs to be be there for the model to be successfully generated
        inputs["input_ids"] = torch.nn.functional.pad(
            torch.zeros((batch_size, sequence_length - 1), dtype=torch.int64), (0, 1), value=32000
        )
    else:
        inputs["position_ids"] = torch.zeros((batch_size, sequence_length), dtype=torch.int64)
        inputs["input_ids"] = torch.zeros((batch_size, sequence_length), dtype=torch.int64)

    return inputs


@Registry.register_dataloader()
def decoder_ort_dataloader(dataset, batch_size, **kwargs):
    return RandomDataLoader(decoder_ort_inputs, batch_size)


# -----------------------------------------------------------------------------
# Quantization calibration
# -----------------------------------------------------------------------------


def tokenize_function(examples):
    # There's a bug that makes the rust-based fast tokenizer hang randomly (probably due to a deadlock),
    # so use the "slow" python one instead
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=False)
    return tokenizer(examples["text"])


class PileDataloader:
    def __init__(self, batch_size=1, seqlen=2048, max_seq_len=2080, sub_folder="train"):
        random.seed(0)
        self.seqlen = seqlen
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dataset = load_dataset("NeelNanda/pile-10k", split=sub_folder)
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def __iter__(self):
        length = len(self.dataset)
        counter = 0

        while counter < length:
            # Pick a random sample from the dataset that has at least 2048 tokens
            sample_index = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[sample_index]["input_ids"]
            while sample.shape[0] <= self.seqlen:
                sample_index = random.randint(0, len(self.dataset) - 1)
                sample = self.dataset[sample_index]["input_ids"]

            # Randomly pick a subsequence of 2048 tokens in the middle of the dataset
            token_start = random.randint(0, sample.shape[0] - self.seqlen - 1)
            token_end = token_start + self.seqlen
            input_ids = sample[token_start:token_end].unsqueeze(0).cpu().numpy().astype("int64")

            initial_position_ids = np.arange(self.seqlen, dtype=np.int64).reshape((1, self.seqlen))
            attention_mask = np.pad(
                np.ones((1, self.seqlen), dtype=np.int64), ((0, 0), (0, self.max_seq_len - self.seqlen))
            )

            initial_inputs = {
                "input_ids": input_ids,
                "position_ids": initial_position_ids,
                "attention_mask": attention_mask,
            }

            for layer_index in range(config.num_layers):
                initial_inputs[f"past_key_values.{layer_index}.key"] = np.zeros(
                    (1, config.num_key_value_heads, self.max_seq_len, config.head_dim), dtype=np.float16
                )
                initial_inputs[f"past_key_values.{layer_index}.value"] = np.zeros(
                    (1, config.num_key_value_heads, self.max_seq_len, config.head_dim), dtype=np.float16
                )

            yield initial_inputs, 0


@Registry.register_dataloader()
def directml_llm_calib_dataloader(dataset, batch_size, **kwargs):
    return PileDataloader(batch_size=batch_size)
