# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaTokenizer

from olive.constants import Framework
from olive.model import PyTorchModel

# -----------------------------------------------------------------------------
# Dummy Inputs
# -----------------------------------------------------------------------------


def get_merged_decoder_with_past_dummy_inputs(model: PyTorchModel):
    """Get dummy inputs for merged decoder model with past_key_values."""
    # Dummy values for export
    batch_size, seq_length, past_seq_length = 2, 8, 0
    return get_merged_sample_with_past_kv_inputs(model, batch_size, seq_length, past_seq_length)


def get_merged_sample_with_past_kv_inputs(
    model: PyTorchModel,
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

    if model_id:
        config = LlamaConfig.from_pretrained(model_id)
    else:
        config = LlamaConfig.from_pretrained(model.hf_config.model_name)

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), dtype=torch.int64)
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, dtype=torch.int64)
    position_ids = get_position_ids(attention_mask, past_seq_len=past_seq_len)
    past_kv = get_past_kv_inputs(config, batch_size, past_seq_len, use_fp16=use_fp16)

    return (input_ids, attention_mask, position_ids, past_kv)


def get_position_ids(attention_mask: torch.Tensor, past_seq_len: int):
    """Get position_ids from attention_mask."""
    # this is generic but in practice we only expect to see two scenarios for (past_seq_len, seq_len)
    # prompt generation: (0, seq_len) -> position_ids = (batch_size, seq_len)
    # token generation: (past_seq_len, 1) -> position_ids = (batch_size, 1)
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


def get_merged_decoder_with_past_io_config(model_name):
    config = LlamaConfig.from_pretrained(model_name)

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
            model=None,
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
                # GQA supports a causal mask by default
                del inputs["attention_mask"]
                inputs["past_sequence_length"] = torch.tensor([self.past_seq_len], dtype=torch.int64)
                inputs = enable_past_present_share_buffer(inputs, self.past_seq_len, self.max_seq_len)

        return (inputs, None)


def dataloader_func_for_merged(data_dir, batch_size, *args, **kwargs):
    """Return data loader for input PyTorch model and ONNX models with past_key_values."""
    model_id = kwargs["model_id"]
    seq_length, past_seq_length, max_seq_length = 8, 0, 2048
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    return RandomDataLoader(
        model_id, batch_size, seq_length, past_seq_length, max_seq_length, model_framework=model_framework
    )


def dataloader_func_for_merged_gqa(data_dir, batch_size, *args, **kwargs):
    """Return data loader for ONNX model + FP16 + GQA."""
    model_id = kwargs["model_id"]
    seq_length, past_seq_length, max_seq_length = 8, 0, 2048
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    return RandomDataLoader(
        model_id,
        batch_size,
        seq_length,
        past_seq_length,
        max_seq_length,
        model_framework=model_framework,
        use_fp16=True,
        use_gqa=True,
    )


# -----------------------------------------------------------------------------
# Inc Smooth Quant Calibration Data Loader
# -----------------------------------------------------------------------------


def inc_cali_dataloader_func(data_dir, batch_size, *args, **kwargs):
    return QuantKVDataLoader(
        hf_model_id="meta-llama/Llama-2-7b-hf",
        dataset_name="NeelNanda/pile-10k",
    )


def inc_cali_merged_dataloader_func(data_dir, batch_size, *args, **kwargs):
    return QuantKVDataLoader(
        hf_model_id="meta-llama/Llama-2-7b-hf",
        dataset_name="NeelNanda/pile-10k",
        merged=True,
    )


class QuantKVDataLoader:
    def __init__(self, hf_model_id: str = "", dataset_name: str = "", pad_max: int = 196, merged: bool = False):
        self.batch_size = 1
        self.pad_max = pad_max
        self.merged = merged

        tokenizer = LlamaTokenizer.from_pretrained(hf_model_id)
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

    def collate_batch(self, batch):
        input_ids_batched = []
        attention_mask_batched = []
        position_ids_batched = []
        labels = []

        for text in batch:
            # Set inputs for model
            input_ids = text["input_ids"]
            attention_mask = torch.ones(len(input_ids))
            position_ids = get_position_ids(attention_mask, past_seq_len=0)
            label = len(input_ids) - 1

            # Pad input data because all model inputs must have same shape
            pad_len = self.pad_max - input_ids.shape[0]
            # pylint: disable=not-callable
            input_ids = F.pad(input_ids, (0, pad_len), value=1)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
            position_ids = F.pad(position_ids, (0, pad_len), value=0)

            input_ids_batched.append(input_ids)
            attention_mask_batched.append(attention_mask)
            position_ids_batched.append(position_ids)
            labels.append(label)

        input_ids_batched = torch.vstack(input_ids_batched)
        attention_mask_batched = torch.vstack(attention_mask_batched)
        position_ids_batched = torch.vstack(position_ids_batched)
        labels = torch.tensor(labels)

        return (input_ids_batched, attention_mask_batched, position_ids_batched), labels

    def __iter__(self):
        for (input_ids, attention_mask, position_ids), labels in self.dataloader:
            # Inputs for decoder_model.onnx
            inputs = {
                "input_ids": input_ids[:, :-1].detach().cpu().numpy().astype(np.int64),
                "attention_mask": attention_mask[:, :-1].detach().cpu().numpy().astype(np.int64),
                "position_ids": position_ids[:, :-1].detach().cpu().numpy().astype(np.int64),
            }
            if self.merged:
                inputs.pop("attention_mask", None)
            label = labels.detach().cpu().numpy()

            # Yield (inputs, label) tuple for Intel's Neural Compressor:
            # https://github.com/intel/neural-compressor/blob/d4baed9ea11614e1f0dc8a1f4f55b73ed3ed585c/neural_compressor/quantization.py#L55-L62
            yield (inputs, label)
