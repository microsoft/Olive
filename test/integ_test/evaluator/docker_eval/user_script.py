# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


def post_process(res):
    return res.argmax(1)


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    dataset = datasets.MNIST(data_dir, transform=ToTensor())
    return torch.utils.data.DataLoader(dataset, batch_size)


def openvino_post_process(res):
    res = next(iter(res.values()))
    return res.argmax(1)


def hf_post_process(res):
    import transformers

    if isinstance(res, transformers.modeling_outputs.SequenceClassifierOutput):
        _, preds = torch.max(res.logits, dim=1)
    else:
        _, preds = torch.max(res, dim=1)
    return preds


def create_hf_dataloader(data_dir, batch_size, *args, **kwargs):
    from datasets import load_dataset
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    dataset = load_dataset("glue", "mrpc", split="validation")

    class BaseData(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            data = {k: v for k, v in self.data[idx].items() if k != "label"}
            return data, self.data[idx]["label"]

    def _map(examples):
        t_input = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True)
        t_input["label"] = examples["label"]
        return t_input

    dataset = dataset.map(
        _map,
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset.set_format(type="torch", output_all_columns=True)
    return torch.utils.data.DataLoader(BaseData(dataset), batch_size)
