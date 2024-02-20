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
from transformers import AutoConfig, AutoTokenizer


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
        if config.model_type == "llava":
            llava_config = AutoConfig.from_pretrained(config.model_id)
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
            new_dict = convert_falcon_weights()
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


def decoder_torch_inputs(model):
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


def decoder_ort_inputs(batch_size):
    past_seq_len = 0
    seq_len = 10
    max_seq_len = past_seq_len + seq_len
    head_size = config.hidden_size // config.num_heads

    inputs = {
        "attention_mask": torch.zeros((batch_size, max_seq_len), dtype=torch.int64),
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


def decoder_ort_data_loader(data_dir, batch_size, *args, **kwargs):
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
    def __init__(self, model_path, batch_size=1, seqlen=2048, max_seq_len=2080, sub_folder="train"):
        random.seed(0)
        self.seqlen = seqlen
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.head_size = config.hidden_size // config.num_heads
        self.dataset = load_dataset("NeelNanda/pile-10k", split=sub_folder)
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.sess = None

    def __iter__(self):
        try:
            while True:
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

                seqlens_k = np.array(0, dtype=np.int32, ndmin=1)
                initial_position_ids = np.arange(self.seqlen, dtype=np.int64).reshape((1, self.seqlen))

                initial_inputs = {
                    "tokens": input_ids,
                    "position_ids": initial_position_ids,
                    "seqlens_k": seqlens_k,
                }

                for layer_index in range(config.num_layers):
                    initial_inputs[f"cache.{layer_index}.key"] = np.zeros(
                        (1, config.num_key_value_heads, self.max_seq_len, self.head_size), dtype=np.float16
                    )
                    initial_inputs[f"cache.{layer_index}.value"] = np.zeros(
                        (1, config.num_key_value_heads, self.max_seq_len, self.head_size), dtype=np.float16
                    )

                yield initial_inputs, 0

        except StopIteration:
            return


def calib_dataloader(data_dir, batch_size, *args, **kwargs):
    model_path = kwargs.pop("model_path")
    return PileDataloader(model_path, batch_size=batch_size)
