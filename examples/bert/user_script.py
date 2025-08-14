# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# pylint: disable=attribute-defined-outside-init, protected-access, ungrouped-imports
# This file is only used by bert_inc_ptq_cpu


import numpy as np
import torch
import torchmetrics
import transformers
from datasets import load_dataset
from datasets.utils import logging as datasets_logging
from neural_compressor.data import DefaultDataLoader
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)

from olive.constants import Framework
from olive.data.registry import Registry
from olive.model import OliveModelHandler

datasets_logging.disable_progress_bar()
datasets_logging.set_verbosity_error()


# -------------------------------------------------------------------------
# Common Dataset
# -------------------------------------------------------------------------


class BertDataset:
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path

        self.task_name = "mrpc"
        self.max_seq_length = 128
        self.data_collator = default_data_collator
        self.padding = "max_length"
        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=2,
            finetuning_task=self.task_name,
            cache_dir=None,
            revision="main",
            use_auth_token=None,
        )
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
        return self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

    def get_train_dataset(self):
        self.train_dataset = self.raw_datasets["train"]
        return self.train_dataset

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


# -------------------------------------------------------------------------
# Post Processing Function for Accuracy Calculation
# -------------------------------------------------------------------------


@Registry.register_post_process()
def bert_post_process(output):
    if isinstance(output, transformers.modeling_outputs.SequenceClassifierOutput):
        _, preds = torch.max(output[0], dim=1)
    else:
        _, preds = torch.max(output, dim=1)
    return preds


# -------------------------------------------------------------------------
# Dataloader for Evaluation and Performance Tuning
# -------------------------------------------------------------------------


@Registry.register_dataset()
def bert_dataset(data_name: str):
    return BertDataset(data_name).get_eval_dataset()


@Registry.register_dataloader()
def bert_dataloader(dataset, batch_size, **kwargs):
    return torch.utils.data.DataLoader(BertDatasetWrapper(dataset), batch_size=batch_size, drop_last=True)


# -------------------------------------------------------------------------
# Calibration Data Reader for Intel® Neural Compressor Quantization
# -------------------------------------------------------------------------


class IncBertDataset:
    """Dataset for Intel® Neural Compressor must implement __iter__ or __getitem__ magic method."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        data = sample
        input_dict = {
            "input_ids": np.array(data["input_ids"]),
            "attention_mask": np.array(data["attention_mask"]),
            "token_type_ids": np.array(data["token_type_ids"]),
        }
        label = data["label"]
        return input_dict, label


@Registry.register_dataset()
def bert_inc_glue_calibration_dataset(data_dir, **kwargs):
    dataset = BertDataset("Intel/bert-base-uncased-mrpc")
    return IncBertDataset(dataset.get_eval_dataset())


@Registry.register_dataloader()
def bert_inc_glue_calibration_dataloader(dataset, batch_size=1, **kwargs):
    return DefaultDataLoader(dataset=dataset, batch_size=batch_size)


# -------------------------------------------------------------------------
# Accuracy Calculation Function
# -------------------------------------------------------------------------


def eval_accuracy(model: OliveModelHandler, device, execution_providers, batch_size=1, **kwargs):
    dataset = bert_dataset("Intel/bert-base-uncased-mrpc")
    dataloader = bert_dataloader(dataset, batch_size)
    preds = []
    target = []
    sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)
    if model.framework == Framework.ONNX:
        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]
        for inputs_i, labels in dataloader:
            if isinstance(inputs_i, dict):
                input_dict = {k: inputs_i[k].tolist() for k in inputs_i}
            else:
                inputs = inputs_i.tolist()
                input_dict = dict(zip(input_names, [inputs]))
            res = model.run_session(sess, input_dict)
            if len(output_names) == 1:
                result = torch.Tensor(res[0])
            else:
                result = torch.Tensor(res)
            outputs = bert_post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.data.tolist())
    elif model.framework == Framework.PYTORCH:
        for inputs, labels in dataloader:
            result = model.run_session(sess, inputs)
            outputs = bert_post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.data.tolist())

    preds_tensor = torch.tensor(preds, dtype=torch.int)
    target_tensor = torch.tensor(target, dtype=torch.int)
    accuracy = torchmetrics.Accuracy(task="binary")
    result = accuracy(preds_tensor, target_tensor)
    return result.item()
