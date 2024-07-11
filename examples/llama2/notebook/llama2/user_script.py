# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import List, Tuple, Union

import torch
from datasets import load_dataset
from transformers import LlamaConfig

from olive.constants import Framework
from olive.data.registry import Registry

# -----------------------------------------------------------------------------
# Dummy Inputs
# -----------------------------------------------------------------------------


def get_merged_sample_with_past_kv_inputs(
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
    config = LlamaConfig.from_pretrained(model_id)

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.int64)
    position_ids = get_position_ids(attention_mask, past_seq_len=past_seq_len)
    position_ids = position_ids.to(torch.int64)
    past_kv = get_past_kv_inputs(config, batch_size, past_seq_len, use_fp16=use_fp16)

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


def get_past_kv_inputs(config: LlamaConfig, batch_size: int, past_seq_len: int, use_fp16: bool):
    """Get past_key_values for all layers.

    Shape of past_key_values is (batch_size, num_heads, past_seq_len, head_size).
    """
    num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_key_value_heads
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    return [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]


def flatten_past_kv_inputs(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Flatten past_key_values to a dict of past_key and past_value. For ONNX model only."""
    past_kv = {}
    # Convert list of past_kv to dict of past_key and past_value
    for i, (past_k, past_v) in enumerate(past_key_values):
        past_kv[f"past_key_values.{i}.key"] = past_k
        past_kv[f"past_key_values.{i}.value"] = past_v
    return past_kv


def enable_past_present_share_buffer(ort_inputs: dict, past_seq_len: int, max_seq_len: int):
    """Enable past-present share buffer for GQA. For ONNX model + FP16 + GQA only."""
    for k, v in ort_inputs.items():
        # Allocate new buffers with max_seq_len for GQA
        if "past_key_values" in k:
            # Copy v (BxSxPxH) into new_v (BxSxMxH)
            batch_size, num_heads, _, head_size = v.shape
            new_v = torch.zeros((batch_size, num_heads, max_seq_len, head_size), dtype=v.dtype)
            new_v[:batch_size, :num_heads, :past_seq_len, :head_size] = v
            ort_inputs[k] = new_v
    return ort_inputs


# -----------------------------------------------------------------------------
# Metric Data Loader
# -----------------------------------------------------------------------------


class RandomDataLoader:
    def __init__(
        self,
        model_id: str,
        batch_size: int,
        seq_len: int,
        past_seq_len: int,
        max_seq_len: int,
        model_framework: str = Framework.PYTORCH,
        use_fp16: bool = False,
        use_gqa: bool = False,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.past_seq_len = past_seq_len
        # TODO(anyone): should we get max_seq_len from config?
        self.max_seq_len = max_seq_len
        self.model_framework = model_framework
        if use_gqa and (model_framework != Framework.ONNX or not use_fp16):
            raise ValueError("GQA is only supported for ONNX model with FP16")
        self.use_fp16 = use_fp16
        self.use_gqa = use_gqa

    def __getitem__(self, idx):
        input_ids, attention_mask, position_ids, past_kv = get_merged_sample_with_past_kv_inputs(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            past_seq_len=self.past_seq_len,
            use_fp16=self.use_fp16,
            model_id=self.model_id,
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_kv,
        }
        if self.model_framework == Framework.ONNX:
            inputs.update(flatten_past_kv_inputs(past_kv))
            del inputs["past_key_values"]

            if self.use_gqa:
                inputs = enable_past_present_share_buffer(inputs, self.past_seq_len, self.max_seq_len)

        return (inputs, None)


def _dataloader_func(**kwargs):
    """Return dataloader for both prompt generation and token generation with/without GQA + FP16."""
    print(f"data loader kwargs: {kwargs}")  # noqa: T201
    batch_size = kwargs["batch_size"]
    model_id = kwargs["model_id"]
    seq_length = kwargs["seq_length"]
    past_seq_length = kwargs["past_seq_length"]
    assert (seq_length >= 1 and past_seq_length == 0) or (seq_length == 1 and past_seq_length >= 1), (
        "Invalid seq_length and past_seq_length. Must be either prompt generation: (seq_length >= 1 and past_seq_length"
        " == 0) or token generation: (seq_length == 1 and past_seq_length >= 1)."
    )
    max_seq_length = kwargs["max_seq_length"]
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    use_fp16 = kwargs.get("use_fp16", False)
    use_gqa = kwargs.get("use_gqa", False)
    return RandomDataLoader(
        model_id,
        batch_size,
        seq_length,
        past_seq_length,
        max_seq_length,
        model_framework=model_framework,
        use_fp16=use_fp16,
        use_gqa=use_gqa,
    )


def dataloader_func_for_merged(data_dir, batch_size, **kwargs):
    """Return data loader for input PyTorch model and ONNX models with past_key_values."""
    return _dataloader_func(batch_size=batch_size, **kwargs)


def dataloader_func_for_merged_gqa(data_dir, batch_size, **kwargs):
    """Return data loader for ONNX model + FP16 + GQA."""
    return _dataloader_func(batch_size=batch_size, use_fp16=True, use_gqa=True, **kwargs)


# -----------------------------------------------------------------------------
#  QLoRA load_dataset component
# -----------------------------------------------------------------------------


@Registry.register_dataset()
def load_tiny_code_dataset(
    data_name: str, split: str, language: str, token: Union[bool, str] = True, trust_remote_code=None
):
    dataset = load_dataset(data_name, split=split, token=token)
    return dataset.filter(lambda x: x["programming_language"] == language)
