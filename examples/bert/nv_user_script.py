# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from datasets import load_dataset  # type: ignore[import]
from datasets.utils import logging as datasets_logging  # type: ignore[import]
from transformers import AutoTokenizer, BertModel  # type: ignore[import]

datasets_logging.disable_progress_bar()
datasets_logging.set_verbosity_error()


def load_pytorch_origin_model(model_path):
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return model


def create_calibration_dataloader(data_dir, batch_size, calib_size, *args, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenization(example):
        return tokenizer(example["text"], padding="max_length", max_length=128, truncation=True)

    dataset = load_dataset("rotten_tomatoes", split="validation[:10%]")
    dataset = dataset.map(tokenization, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])

    return torch.utils.data.DataLoader(dataset.select(range(calib_size)), batch_size=batch_size, drop_last=True)
