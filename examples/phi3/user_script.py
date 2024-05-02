# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from argparse import Namespace
from itertools import chain
from typing import List, Union
import torch
from datasets import load_dataset
from transformers import AutoConfig, PretrainedConfig
from olive.data.registry import Registry

from olive.model import PyTorchModelHandler

# -----------------------------------------------------------------------------
# Dummy Inputs
# -----------------------------------------------------------------------------


def get_merged_decoder_with_past_dummy_inputs(model: PyTorchModelHandler):
    """Get dummy inputs for merged decoder model with past_key_values."""
    # Dummy values for export
    batch_size, seq_length, past_seq_length = 2, 8, 0
    return get_merged_sample_with_past_kv_inputs(model, batch_size, seq_length, past_seq_length)


def get_merged_sample_with_past_kv_inputs(
    model: PyTorchModelHandler,
    batch_size: int,
    seq_len: int,
    past_seq_len: int,
    use_fp16: bool = False,
    model_id: str = "",
):
    """Get inputs for forward pass with past_key_values.

    This is for the "merged" model which can be used for both prompt generation and token generation.
    For prompt generation, past_seq_len = 0 and seq_len >= 1. past_kv is a list of tuples of empty tensors.
    For token generation, past_seq_len >= 1 and seq_len = 1.

    Shape of outputs:
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, past_seq_len + seq_len)
        position_ids: (batch_size, seq_len)
        past_key: (batch_size, num_heads, past_seq_len, head_size)
        past_value: (batch_size, num_heads, past_seq_len, head_size)
    """
    # Note: No need for separate function for legacy prompt and token generation
    # prompt generation (get_sample_inputs):
    #   past_seq_len = 0, seq_len >= 1, use_gqa = False, use_fp16 = False
    #   and remove past_kv from the output
    # token generation (get_sample_with_past_kv_inputs):
    #   past_seq_len >= 1, seq_len = 1, use_gqa = False, use_fp16 = False
    # By using a single function with no default values, we can avoid confusion and are deliberate about the sizes
    # can instead write dummy input functions like 'get_merged_decoder_with_past_dummy_inputs' if needed
    if not model:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    elif model.hf_config is not None:
        config = model.get_hf_model_config() if model else AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    else:
        # Using Namespace class to access dict items like class attributes
        config = Namespace(**model.model_attributes)
    world_size = config.world_size if hasattr(config, "world_size") else 1
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.int64)
    position_ids = get_position_ids(attention_mask, past_seq_len=past_seq_len)
    past_kv = get_past_kv_inputs(config, batch_size, past_seq_len, use_fp16=use_fp16, world_size=world_size)

    return (input_ids, attention_mask, position_ids, past_kv)


def get_position_ids(attention_mask: torch.Tensor, past_seq_len: int):
    """Get position_ids from attention_mask."""
    # this is generic but in practice we only expect to see two scenarios for (past_seq_len, seq_len)
    # prompt generation: (0, seq_len) -> position_ids = (batch_size, seq_len)
    # token generation: (past_seq_len, 1) -> position_ids = (batch_size, 1)
    # Note: The merged model only works in these two scenarios
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids[:, past_seq_len:]


def get_past_kv_inputs(
    config: PretrainedConfig, batch_size: int, past_seq_len: int, use_fp16: bool, world_size: int = 1
):
    """Get past_key_values for all layers.

    Shape of past_key_values is (batch_size, num_heads, past_seq_len, head_size).
    """
    num_heads = config.num_key_value_heads // world_size
    head_size = config.hidden_size // config.num_attention_heads
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    return [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]


# -----------------------------------------------------------------------------
# Conversion Arguments (Inputs, Outputs, Dynamic Axes)
# -----------------------------------------------------------------------------


def get_merged_model_dynamic_axes(input_names: List[str], output_names: List[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"input_ids", "position_ids"}:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "attention_mask":
            # shape is (batch_size, past_sequence_length + sequence_length) = (batch_size, total_sequence_length)
            # for prompt generation, past_sequence_length = 0
            # for token generation, sequence_length = 1
            dynamic_axes[name] = {0: "batch_size", 1: "total_sequence_length"}
        elif "past" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + sequence_length, head_size)
            #  = (batch_size, num_heads, total_sequence_length, head_size)
            # for prompt generation, past_sequence_length = 0
            # for token generation, sequence_length = 1
            dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}
        else:
            raise ValueError("Unknown input or output name found")
    return dynamic_axes


def get_merged_decoder_with_past_io_config(model: PyTorchModelHandler):
    if model.hf_config is not None:
        config = model.get_hf_model_config()
    else:
        # Using Namespace class to access dict items like class attributes
        config = Namespace(**model.model_attributes)

    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        *list(
            chain.from_iterable(
                (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(config.num_hidden_layers)
            )
        ),
    ]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(config.num_hidden_layers))),
    ]
    dynamic_axes = get_merged_model_dynamic_axes(input_names, output_names)
    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
    }


# -----------------------------------------------------------------------------
#  QLoRA load_dataset component
# -----------------------------------------------------------------------------


@Registry.register_dataset()
def load_tiny_code_dataset(data_name: str, split: str, language: str, token: Union[bool, str] = True):
    dataset = load_dataset(data_name, split=split, token=token)
    return dataset.filter(lambda x: x["programming_language"] == language)
