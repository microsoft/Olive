# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from datasets.utils import logging as datasets_logging  # type: ignore[import]
from transformers import AutoTokenizer

from olive.data.registry import Registry

datasets_logging.disable_progress_bar()
datasets_logging.set_verbosity_error()


@Registry.register_dataloader("nvmo_calibration_dataloader")
def create_calibration_dataloader(dataset, batch_size, calib_size=64, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenization(example):
        return tokenizer(example["text"], padding="max_length", max_length=128, truncation=True)

    dataset = dataset.map(tokenization, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])

    return torch.utils.data.DataLoader(dataset.select(range(calib_size)), batch_size=batch_size, drop_last=True)
