# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
import transformers
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from onnxruntime.quantization.calibrate import CalibrationDataReader
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from neural_compressor.data import DefaultDataLoader
import numpy as np

disable_progress_bar()


def create_input_tensors():
    return {
        "input_ids": torch.ones(1, 128, dtype=torch.int64),
        "attention_mask": torch.ones(1, 128, dtype=torch.int64),
        "token_type_ids": torch.ones(1, 128, dtype=torch.int64),
    }


class BertDataset:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path

        self.task_name = "mrpc"
        self.max_seq_length = 128
        self.padding = "max_length"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            cache_dir=None,
            use_fast=True,
            revision="main",
            use_auth_token=None,
        )
        self.setup_dataset()

    def setup_dataset(self):
        self.raw_datasets = load_dataset("glue", self.task_name, cache_dir=None)
        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    def preprocess_function(self, examples):
        sentence1_key, sentence2_key = ("sentence1", "sentence2")
        args = (examples[sentence1_key], examples[sentence2_key])
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
        return result

    def get_eval_dataset(self):
        self.eval_dataset = self.raw_datasets["validation"]
        return self.eval_dataset


class BertDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        data = sample
        input_dict = {
            "input_ids": torch.tensor(data["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(data["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(data["token_type_ids"], dtype=torch.long),
        }
        label = data["label"]
        return input_dict, label


def post_process(output):
    if isinstance(output, transformers.modeling_outputs.SequenceClassifierOutput):
        _, preds = torch.max(output[0], dim=1)
    else:
        _, preds = torch.max(output, dim=1)
    return preds


def create_dataloader(data_dir, batchsize):
    bert_dataset = BertDataset("Intel/bert-base-uncased-mrpc")
    eval_dataloader = torch.utils.data.DataLoader(
        BertDatasetWrapper(bert_dataset.get_eval_dataset()), batch_size=batchsize, drop_last=True
    )
    return eval_dataloader


def load_pytorch_origin_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained("Intel/bert-base-uncased-mrpc")
    model.eval()
    return model

class IncBertDataset:
    """
    Dataset for Intel Neural Compressor must implement __iter__ or __getitem__ magic method.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        data = sample
        input_dict = {
            "input_ids": np.array(data["input_ids"], dtype=np.long),
            "attention_mask": np.array(data["attention_mask"], dtype=np.long),
            "token_type_ids": np.array(data["token_type_ids"], dtype=np.long),
        }
        label = data["label"]
        return input_dict, label
    
def inc_glue_calibration_reader(data_dir, batch_size=1):
    bert_dataset = BertDataset("Intel/bert-base-uncased-mrpc")
    bert_dataset = IncBertDataset(bert_dataset.get_eval_dataset())
    calib_dataloader = DefaultDataLoader(dataset=bert_dataset, batch_size=batch_size)
    return calib_dataloader
