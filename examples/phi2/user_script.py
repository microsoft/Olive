# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from olive.constants import Framework

if TYPE_CHECKING:
    from transformers import PhiConfig

model_id = "microsoft/phi-2"
config: "PhiConfig" = AutoConfig.from_pretrained(model_id, trust_remote_code=True)


# TODO(myguo): the following code is copied from
# https://huggingface.co/microsoft/phi-2/blob/main/modeling_phi.py
# need remove the definition and forward patch logic after the official phi2 support the list of past_key_values
@dataclass
class InferenceParams:
    """Inference parameters passed to model to efficiently calculate and store context during inference.

    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py.

    Args:
        max_seqlen: Maximum sequence length.
        max_batch_size: Maximum batch size.
        seqlen_offset: Sequence length offset.
        batch_size_offset: Batch size offset.
        key_value_memory_dict: Key value memory dictionary.
        lengths_per_sample: Lengths per sample.
    """

    max_seqlen: int = field(metadata={"help": "Maximum sequence length."})

    max_batch_size: int = field(metadata={"help": "Maximum batch size."})

    seqlen_offset: int = field(default=0, metadata={"help": "Sequence length offset."})

    batch_size_offset: int = field(default=0, metadata={"help": "Batch size offset."})

    key_value_memory_dict: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Key value memory dictionary."}
    )

    lengths_per_sample: torch.Tensor = field(default=None, metadata={"help": "Lengths per sample."})


def post_process(result, num_layer):
    present = []
    for i in range(num_layer):
        present.append(result.past_key_values.key_value_memory_dict[i])
    return (result.logits, tuple(present))


def patched_forward(forward):
    def inner_forward(input_ids, attention_mask, past_key_values):
        kv_mem_dict = {}
        for i, kv in enumerate(past_key_values):
            kv_mem_dict[i] = kv

        inference_params = InferenceParams(
            max_seqlen=config.n_positions,
            max_batch_size=input_ids.shape[0],
            seqlen_offset=0,
            batch_size_offset=0,
            key_value_memory_dict=kv_mem_dict,
            lengths_per_sample=None,
        )
        result = forward(input_ids, attention_mask=attention_mask, past_key_values=inference_params)
        return post_process(result, config.num_hidden_layers)

    return inner_forward


def model_loader(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
    model.forward = patched_forward(model.forward)
    return model


def dummy_inputs(model):
    """Get dummy inputs for merged decoder model with past_key_values."""
    batch_size, sequence_length, past_sequence_length = 2, 8, 0
    max_sequence_length = 512

    return get_merged_sample_with_past_kv_inputs(
        config,
        torch.device("cpu"),
        batch_size,
        sequence_length,
        past_sequence_length,
        max_sequence_length,
        use_fp16=False,
        use_gqa=False,
        engine="pt",
        return_dict=True,
        world_size=1,
    )


def get_io_config(model):
    input_names = [
        "input_ids",
        "attention_mask",
        *list(chain.from_iterable((f"past_key_values.{i}",) for i in range(config.num_hidden_layers))),
    ]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present_key_values.{i}",) for i in range(config.num_hidden_layers))),
    ]
    dynamic_axes = get_merged_model_dynamic_axes(input_names, output_names)
    return {
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    sequence_length, past_sequence_length = 8, 0
    max_sequence_length = 512
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    engine = "ort" if model_framework == Framework.ONNX else "pt"

    return RandomDataLoader(batch_size, sequence_length, past_sequence_length, max_sequence_length, engine=engine)


class RandomDataLoader:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        past_seq_len: int,
        max_seq_len: int,
        engine: str = "pt",
        use_fp16: bool = False,
        use_gqa: bool = False,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.past_seq_len = past_seq_len
        self.max_seq_len = max_seq_len
        self.engine = engine
        if use_gqa and (engine != "ort" or not use_fp16):
            raise ValueError("GQA is only supported for ONNX model with FP16")
        self.use_fp16 = use_fp16
        self.use_gqa = use_gqa

    def __getitem__(self, idx):
        inputs = get_merged_sample_with_past_kv_inputs(
            config,
            device=torch.device("cpu"),
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            past_seq_len=self.past_seq_len,
            max_seq_len=self.max_seq_len,
            use_fp16=self.use_fp16,
            use_gqa=self.use_gqa,
            engine=self.engine,
            return_dict=True,
        )
        return (inputs, None)


def get_position_ids(attention_mask: torch.Tensor, past_seq_len: int):
    """Get position_ids from attention_mask."""
    # this is generic but in practice we only expect to see two scenarios for (past_seq_len, seq_len)
    # prompt generation: (0, seq_len) -> position_ids = (batch_size, seq_len)
    # token generation: (past_seq_len, 1) -> position_ids = (batch_size, 1)
    # Note: The merged model only works in these two scenarios
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids[:, past_seq_len:]


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
            dynamic_axes[name] = {0: "batch_size", 1: "max_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + sequence_length, head_size)
            # = (batch_size, num_heads, total_sequence_length, head_size)
            # for prompt generation, past_sequence_length = 0
            # for token generation, sequence_length = 1
            dynamic_axes[name] = {0: "batch_size", 1: "max_sequence_length"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes


# Inputs for all passes with past_key_values
#   input_ids: (batch_size, sequence_length)
#   attention_mask: (batch_size, past_sequence_length + sequence_length)
#   past_kv: (batch_size, max_sequence_length, 2, num_heads, head_size)
def get_merged_sample_with_past_kv_inputs(
    config: AutoConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    past_seq_len: int,
    max_seq_len: int,
    use_fp16: bool = False,
    use_gqa: bool = False,
    engine: str = "pt",
    return_dict: bool = False,
    world_size: int = 1,
):
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.int64)
    # position_ids is of shape (batch_size, seq_len) for prompt generation, (batch_size, 1) for token generation
    # position_ids = get_position_ids(attention_mask, use_past_kv=(past_seq_len != 0))
    past_kv = get_past_kv_inputs(config, batch_size, max_seq_len, use_fp16, world_size=world_size)

    # Convert inputs to NumPy (for ORT) or send to device (for PyTorch)
    input_ids = input_ids.numpy() if engine == "ort" else input_ids.to(device)
    attention_mask = attention_mask.numpy() if engine == "ort" else attention_mask.to(device)
    # position_ids = position_ids.numpy() if engine == "ort" else position_ids.to(device)

    # ruff: noqa: C417
    past_kv = flatten_past_kv_inputs(past_kv) if engine == "ort" else list(map(lambda kv: (kv[0].to(device)), past_kv))

    if not return_dict:
        # For export
        assert isinstance(past_kv, list)
        return (input_ids, past_kv, attention_mask)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if engine == "ort":
        assert isinstance(past_kv, dict)
        inputs.update(past_kv)

        if use_gqa:
            inputs = enable_past_present_share_buffer(inputs, past_seq_len, max_seq_len)

    else:
        assert isinstance(past_kv, list)
        inputs["past_key_values"] = past_kv

    return inputs


def flatten_past_kv_inputs(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Flatten past_key_values to a dict of past_key and past_value. For ONNX model only."""
    past_kv = {}
    # Convert list of past_kv to dict of past_key and past_value
    for i, value in enumerate(past_key_values):
        past_kv[f"past_key_values.{i}"] = value[0].numpy()
    return past_kv


def enable_past_present_share_buffer(ort_inputs: dict, past_seq_len: int, max_seq_len: int):
    for k, v in ort_inputs.items():
        # Allocate new buffers with max_sequence_length for GQA
        if "cache" in k or "past_key_values" in k:
            # Copy v (BxSxPxH) into new_v (BxSxMxH)
            batch_size, num_heads, _, head_size = v.shape
            new_v = np.zeros((batch_size, num_heads, max_seq_len, head_size), dtype=v.dtype)
            new_v[:batch_size, :num_heads, :past_seq_len, :head_size] = v
            ort_inputs[k] = new_v
    return ort_inputs


def get_past_kv_inputs(config: AutoConfig, batch_size: int, past_seq_len: int, use_fp16: bool, world_size: int = 1):
    num_heads, head_size = config.n_head // world_size, config.n_embd // config.n_head
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    return [
        (torch.rand(batch_size, past_seq_len, 2, num_heads, head_size, dtype=torch_dtype),)
        for _ in range(config.num_hidden_layers)
    ]
