# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import random
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, LlamaTokenizer

from olive.data.registry import Registry

# pylint: disable=redefined-outer-name

model_id = "mistralai/Mistral-7B-v0.1"
config = AutoConfig.from_pretrained(model_id)


@Registry.register_dataloader()
def mistralai_calib_dataloader(data_dir, batch_size, *args, **kwargs):
    model_path = kwargs.pop("model_path")
    return PileDataloader(model_path, batch_size=batch_size)


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    dataloader = PileDataloader(model_id, batch_size=batch_size, seq_len=32, past_seq_len=32, sub_folder="train")
    for data, label in dataloader:
        d = {name: to_numpy(inp_data) for name, inp_data in data.items()}
        yield d, label


def tokenize_function(examples):
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    return tokenizer(examples["text"])


class PileDataloader:
    def __init__(self, model_path, batch_size=1, seq_len=32, past_seq_len=32, sub_folder="train"):
        random.seed(0)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.past_seq_len = past_seq_len

        dataset = load_dataset("NeelNanda/pile-10k", split=sub_folder)
        self.dataset = dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def __iter__(self):
        if all(d["input_ids"].shape[0] <= self.seq_len for d in self.dataset):
            raise ValueError(f"All inputs are less than seq_len: {self.seq_len}")

        length = len(self.dataset)
        counter = 0

        while counter < length:
            while True:
                i = random.randint(0, len(self.dataset) - 1)
                trainenc = self.dataset[i]
                if trainenc["input_ids"].shape[0] > self.seq_len:
                    break
                print(f"{trainenc['input_ids'].shape[0]} is less than {self.seq_len}")  # noqa: T201
            i = random.randint(0, trainenc["input_ids"].shape[0] - self.seq_len - 1)
            j = i + self.seq_len
            inp = trainenc["input_ids"][i:j].unsqueeze(0)

            attention_mask = torch.ones(self.batch_size, self.past_seq_len + self.seq_len, dtype=torch.int64)
            position_ids = get_position_ids(attention_mask, past_seq_len=self.past_seq_len)

            inputs = {
                "input_ids": inp.detach().cpu().numpy().astype("int64"),
                "attention_mask": attention_mask.detach().cpu().numpy().astype("int64"),
                "position_ids": position_ids.detach().cpu().numpy().astype("int64"),
            }

            past_kv = get_past_kv_inputs(config, self.batch_size, self.past_seq_len, use_fp16=False, world_size=1)
            inputs.update(flatten_past_kv_inputs(past_kv))

            counter += 1
            yield inputs, 0


def get_position_ids(attention_mask: torch.Tensor, past_seq_len: int):
    """Get position_ids from attention_mask."""
    # this is generic but in practice we only expect to see two scenarios for (past_seq_len, seq_len)
    # prompt generation: (0, seq_len) -> position_ids = (batch_size, seq_len)
    # token generation: (past_seq_len, 1) -> position_ids = (batch_size, 1)
    # Note: The merged model only works in these two scenarios
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids[:, past_seq_len:]


def get_past_kv_inputs(config, batch_size: int, past_seq_len: int, use_fp16: bool, world_size: int = 1):
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


def flatten_past_kv_inputs(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Flatten past_key_values to a dict of past_key and past_value. For ONNX model only."""
    past_kv = {}
    # Convert list of past_kv to dict of past_key and past_value
    for i, (past_k, past_v) in enumerate(past_key_values):
        past_kv[f"past_key_values.{i}.key"] = past_k
        past_kv[f"past_key_values.{i}.value"] = past_v
    return past_kv


def to_numpy(data):
    """Convert to numpy ndarrays."""
    if not isinstance(data, np.ndarray):
        if isinstance(data, torch.Tensor):
            if data.dtype is torch.bfloat16:  # pragma: no cover
                return data.detach().cpu().to(torch.float32).numpy()
            if data.dtype is torch.chalf:  # pragma: no cover
                return data.detach().cpu().to(torch.cfloat).numpy()
            return data.detach().cpu().numpy()
        else:
            return np.array(data)
    else:
        return data
