# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, sequence_length, attention_mask_sequence_length, past_sequence_length, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.attention_mask_sequence_length = attention_mask_sequence_length
        self.past_sequence_length = past_sequence_length
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.sequence_length, self.attention_mask_sequence_length, self.past_sequence_length, self.torch_dtype), label


def dolly_v2_inputs(batch_size, sequence_length, attention_mask_sequence_length, past_sequence_length, torch_dtype):
    inputs = {
        "input_ids": torch.randint(10, (batch_size, sequence_length), dtype=torch.int64),
        "attention_mask": torch.randint(10, (batch_size, attention_mask_sequence_length), dtype=torch.int64),
    }

    for layer_index in range(32):
        inputs[f"past_key_values.{layer_index}.key"] = torch.rand((batch_size, 32, past_sequence_length, 128), dtype=torch_dtype)
        inputs[f"past_key_values.{layer_index}.value"] = torch.rand((batch_size, 32, past_sequence_length, 128), dtype=torch_dtype)

    inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)

    return inputs


def dolly_v2_load(model_name):
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b", torch_dtype=torch.float16)
    return model


def dolly_v2_conversion_inputs(model):
    return tuple(dolly_v2_inputs(1, 2, 64, 1, torch.float32).values())


def dolly_v2_data_loader(data_dir, batch_size, sequence_length, attention_mask_sequence_length, past_sequence_length):
    return RandomDataLoader(dolly_v2_inputs, batch_size, sequence_length, attention_mask_sequence_length, past_sequence_length, torch.float16)
