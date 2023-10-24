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


def get_position_ids(attention_mask: torch.Tensor, use_past_kv: bool):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if use_past_kv:
        position_ids = position_ids[:, -1].unsqueeze(-1)
    return position_ids


def get_decoder_inputs(model, batch_size=2, seq_len=100, model_id=""):
    device = torch.device("cpu")
    if model_id:
        config = LlamaConfig.from_pretrained(model_id)
    else:
        config = LlamaConfig.from_pretrained(model.hf_config.model_name)

    input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, seq_len), device=device, dtype=torch.int64
    )
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.int64)
    # position_ids is of shape (batch_size, seq_len)
    position_ids = get_position_ids(attention_mask, use_past_kv=False)

    return (input_ids, attention_mask, position_ids)


def get_decoder_with_past_kv_inputs(model, batch_size=2, seq_len=1, past_seq_len=100, use_fp16=False, model_id=""):
    if model_id:
        config = LlamaConfig.from_pretrained(model_id)
    else:
        config = LlamaConfig.from_pretrained(model.hf_config.model_name)

    device = torch.device("cpu")

    input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, seq_len), device=device, dtype=torch.int64
    )
    attention_mask = torch.ones(batch_size, past_seq_len + seq_len, device=device, dtype=torch.int64)
    # position_ids is of shape (batch_size, 1)
    position_ids = get_position_ids(attention_mask, use_past_kv=True)
    past_key_values = get_sample_past_kv_inputs(config, device, batch_size, past_seq_len, use_fp16=use_fp16)

    return (input_ids, attention_mask, position_ids, past_key_values)


def get_merged_decoder_with_past_kv_inputs(model, batch_size=2, seq_len=8, past_seq_len=0, use_fp16=False, model_id=""):
    input_ids, attention_mask, position_ids, past_key_values = get_decoder_with_past_kv_inputs(
        model, batch_size, seq_len, past_seq_len, use_fp16, model_id
    )
    # position_ids is of shape (batch_size, seq_len) for prompt generation, (batch_size, 1) for token generation
    position_ids = get_position_ids(attention_mask, use_past_kv=(past_seq_len != 0))

    return input_ids, attention_mask, position_ids, past_key_values


def get_sample_past_kv_inputs(
    config: LlamaConfig, device: torch.device, batch_size: int, past_seq_len: int, use_fp16: bool
):
    num_heads, head_size = config.num_key_value_heads, config.hidden_size // config.num_key_value_heads
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    return [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]


def flatten_past_kv_inputs(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]], use_fp16: bool):
    past_kv = {}
    np_dtype = np.float16 if use_fp16 else np.float32
    # Convert list of past_kv to dict of past_key and past_value
    for i, (past_k, past_v) in enumerate(past_key_values):
        past_kv[f"past_key_values.{i}.key"] = past_k.detach().cpu().numpy().astype(np_dtype)
        past_kv[f"past_key_values.{i}.value"] = past_v.detach().cpu().numpy().astype(np_dtype)
    return past_kv


def get_model_dynamic_axes(input_names: List[str], output_names: List[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in input_names:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "sequence_length"}
        else:
            raise ValueError("Unknown input or output name found")
    return dynamic_axes


def get_decoder_io_config(model_name, merged=False):
    config = LlamaConfig.from_pretrained(model_name)

    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = [
        "logits",
        *list(chain.from_iterable((f"present.{i}.key", f"present.{i}.value") for i in range(config.num_hidden_layers))),
    ]
    dynamic_axes = get_model_dynamic_axes(input_names, output_names)
    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
    }


def get_model_with_past_kv_dynamic_axes(input_names: List[str], output_names: List[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"input_ids", "position_ids"}:
            # shape is (batch_size, 1)
            dynamic_axes[name] = {0: "batch_size"}
        elif name == "attention_mask":
            # shape is (batch_size, past_sequence_length + 1)
            dynamic_axes[name] = {0: "batch_size", 1: "past_sequence_length + 1"}
        elif "past" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, 1, vocab_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + 1, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length + 1"}
        else:
            raise ValueError("Unknown input or output name found")
    return dynamic_axes


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


def get_decoder_with_past_io_config(model_name):
    config = LlamaConfig.from_pretrained(model_name)
    io_config = get_decoder_io_config(model_name)

    io_config["input_names"].extend(
        list(
            chain.from_iterable(
                (f"past_key_values.{i}.key", f"past_key_values.{i}.value") for i in range(config.num_hidden_layers)
            )
        )
    )
    io_config["dynamic_axes"] = get_model_with_past_kv_dynamic_axes(io_config["input_names"], io_config["output_names"])
    return io_config


def get_merged_decoder_with_past_io_config(model_name):
    io_config = get_decoder_with_past_io_config(model_name)
    io_config["dynamic_axes"] = get_merged_model_dynamic_axes(io_config["input_names"], io_config["output_names"])
    return io_config


class RandomDataLoader:
    def __init__(
        self, create_inputs_func, batch_size, torch_dtype, model_framework=Framework.PYTORCH, onnx_merged=False
    ):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.model_framework = model_framework
        self.onnx_merged = onnx_merged

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype, self.model_framework, self.onnx_merged), label


def dummy_inputs_for_latency(batch_size, torch_dtype, model_framework=Framework.PYTORCH, onnx_merged=False):
    model_id = "meta-llama/Llama-2-7b-hf"
    if onnx_merged:
        input_ids, attention_mask, position_ids, pkv = get_merged_decoder_with_past_kv_inputs(
            model=None, model_id=model_id
        )
    else:
        input_ids, attention_mask, position_ids, pkv = get_decoder_with_past_kv_inputs(model=None, model_id=model_id)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": pkv,
    }
    if model_framework == Framework.ONNX:
        inputs.update(flatten_past_kv_inputs(pkv, use_fp16=torch_dtype == torch.float16))
        inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)
        del inputs["past_key_values"]
    else:
        inputs["use_cache"] = True

    # TODO(trajep): add past_sequence_length to inputs for GQA
    return inputs


def dataloader_func(data_dir, batch_size, *args, **kwargs):
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    return RandomDataLoader(dummy_inputs_for_latency, batch_size, torch.float16, model_framework)


def dataloader_func_for_merged(data_dir, batch_size, *args, **kwargs):
    # TODO(trajep): after optimization, the model's input will be different
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    return RandomDataLoader(dummy_inputs_for_latency, batch_size, torch.float16, model_framework, True)


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
            position_ids = get_position_ids(attention_mask, use_past_kv=False)
            label = len(input_ids) - 1

            # Pad input data because all model inputs must have same shape
            pad_len = self.pad_max - input_ids.shape[0]
            input_ids = F.pad(input_ids, (0, pad_len), value=1)  # pylint: disable=not-callable
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)  # pylint: disable=not-callable
            position_ids = F.pad(position_ids, (0, pad_len), value=0)  # pylint: disable=not-callable

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
        try:
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

        except StopIteration:
            return
