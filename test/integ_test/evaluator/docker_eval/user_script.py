# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from olive.data.registry import Registry


@Registry.register_post_process()
def mnist_post_process_for_docker_eval(res):
    return res.argmax(1)


@Registry.register_post_process()
def mnist_post_process_openvino_for_docker_eval(res):
    res = next(iter(res))
    return [res.argmax()]


@Registry.register_dataset()
def mnist_dataset_for_docker_eval(data_dir):
    return datasets.MNIST(data_dir, transform=ToTensor())


@Registry.register_post_process()
def mnist_post_process_hf_for_docker_eval(res):
    import transformers

    if isinstance(res, transformers.modeling_outputs.SequenceClassifierOutput):
        _, preds = torch.max(res.logits, dim=1)
    else:
        _, preds = torch.max(res, dim=1)
    return preds


@Registry.register_dataset()
def tiny_bert_dataset_for_docker_eval(data_dir, *args, **kwargs):
    from datasets import load_dataset
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-BertForSequenceClassification")
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
    return BaseData(dataset)
