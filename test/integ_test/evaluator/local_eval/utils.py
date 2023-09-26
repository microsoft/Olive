# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path
from test.integ_test.utils import download_azure_blob
from zipfile import ZipFile

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType


def get_directories():
    current_dir = Path(__file__).resolve().parent

    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return models_dir, data_dir


models_dir, data_dir = get_directories()


def post_process(res):
    return res.argmax(1)


def openvino_post_process(res):
    res = list(res.values())[0]
    return res.argmax(1)


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=ToTensor())
    return torch.utils.data.DataLoader(dataset, batch_size)


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


def get_accuracy_metric(post_process, dataloader=create_dataloader):
    accuracy_metric_config = {
        "post_processing_func": post_process,
        "data_dir": data_dir,
        "dataloader_func": dataloader,
    }
    sub_types = [{"name": AccuracySubType.ACCURACY_SCORE}]
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=sub_types,
        user_config=accuracy_metric_config,
    )


def get_latency_metric(dataloader=create_dataloader):
    latency_metric_config = {
        "data_dir": data_dir,
        "dataloader_func": dataloader,
    }
    sub_types = [{"name": LatencySubType.AVG}]
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=sub_types,
        user_config=latency_metric_config,
    )


def get_hf_accuracy_metric(post_process=hf_post_process, dataloader=create_hf_dataloader):
    return get_accuracy_metric(post_process, dataloader)


def get_hf_latency_metric(dataloader=create_hf_dataloader):
    return get_latency_metric(dataloader)


def get_pytorch_model():
    download_path = models_dir / "model.pt"
    pytorch_model_config = {
        "container": "olivetest",
        "blob": "models/model.pt",
        "download_path": download_path,
    }
    download_azure_blob(**pytorch_model_config)
    return {"model_path": str(download_path)}


def get_huggingface_model():
    return {"hf_config": {"model_class": "AutoModelForSequenceClassification", "model_name": "prajjwal1/bert-tiny"}}


def get_onnx_model():
    download_path = models_dir / "model.onnx"
    onnx_model_config = {
        "container": "olivetest",
        "blob": "models/model.onnx",
        "download_path": download_path,
    }
    download_azure_blob(**onnx_model_config)
    return {"model_path": str(download_path)}


def get_openvino_model():
    download_path = models_dir / "openvino.zip"
    openvino_model_config = {
        "container": "olivetest",
        "blob": "models/openvino.zip",
        "download_path": download_path,
    }
    download_azure_blob(**openvino_model_config)
    with ZipFile(download_path) as zip_ref:
        zip_ref.extractall(models_dir)
    return {"model_path": str(models_dir / "openvino")}


def delete_directories():
    shutil.rmtree(data_dir)
    shutil.rmtree(models_dir)
