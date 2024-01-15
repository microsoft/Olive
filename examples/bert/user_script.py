# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy

import numpy as np
import torch
import torchmetrics
import transformers
from datasets import load_dataset, load_metric
from datasets.utils import logging as datasets_logging
from neural_compressor.data import DefaultDataLoader
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from olive.constants import Framework
from olive.model import OliveModelHandler

datasets_logging.disable_progress_bar()
datasets_logging.set_verbosity_error()

# pylint: disable=attribute-defined-outside-init, protected-access
# This file is only used by bert_inc_ptq_cpu, bert_qat_customized_train_loop_cpu

# -------------------------------------------------------------------------
# Model Loader
# -------------------------------------------------------------------------


def load_pytorch_origin_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained("Intel/bert-base-uncased-mrpc")
    model.eval()
    return model


# -------------------------------------------------------------------------
# Dummy Input for ONNX Export
# -------------------------------------------------------------------------


def create_input_tensors(model):
    return {
        "input_ids": torch.ones(1, 128, dtype=torch.int64),
        "attention_mask": torch.ones(1, 128, dtype=torch.int64),
        "token_type_ids": torch.ones(1, 128, dtype=torch.int64),
    }


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


def post_process(output):
    if isinstance(output, transformers.modeling_outputs.SequenceClassifierOutput):
        _, preds = torch.max(output[0], dim=1)
    else:
        _, preds = torch.max(output, dim=1)
    return preds


# -------------------------------------------------------------------------
# Dataloader for Evaluation and Performance Tuning
# -------------------------------------------------------------------------


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    bert_dataset = BertDataset("Intel/bert-base-uncased-mrpc")
    return torch.utils.data.DataLoader(
        BertDatasetWrapper(bert_dataset.get_eval_dataset()), batch_size=batchsize, drop_last=True
    )


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


def inc_glue_calibration_reader(data_dir, batch_size, *args, **kwargs):
    bert_dataset = BertDataset("Intel/bert-base-uncased-mrpc")
    bert_dataset = IncBertDataset(bert_dataset.get_eval_dataset())
    return DefaultDataLoader(dataset=bert_dataset, batch_size=batch_size)


# -------------------------------------------------------------------------
# Accuracy Calculation Function
# -------------------------------------------------------------------------


def eval_accuracy(model: OliveModelHandler, data_dir, batch_size, device, execution_providers):
    dataloader = create_dataloader(data_dir, batch_size)
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
            res = sess.run(input_feed=input_dict, output_names=None)
            if len(output_names) == 1:
                result = torch.Tensor(res[0])
            else:
                result = torch.Tensor(res)
            outputs = post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.data.tolist())
    elif model.framework == Framework.PYTORCH:
        for inputs, labels in dataloader:
            if isinstance(inputs, dict):
                result = sess(**inputs)
            else:
                result = sess(inputs)
            outputs = post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.data.tolist())

    preds_tensor = torch.tensor(preds, dtype=torch.int)
    target_tensor = torch.tensor(target, dtype=torch.int)
    accuracy = torchmetrics.Accuracy(task="binary")
    result = accuracy(preds_tensor, target_tensor)
    return result.item()


# -------------------------------------------------------------------------
# Trainer for Quantization Aware Training
# -------------------------------------------------------------------------


def training_loop_func(model):
    set_seed(42)

    training_args = TrainingArguments("bert_qat")
    training_args._n_gpu = 0
    training_args.learning_rate = 2e-5
    training_args.do_eval = True
    training_args.do_train = True
    training_args.per_device_train_batch_size = 8
    training_args.per_device_eval_batch_size = 8
    training_args.num_train_epochs = 2
    training_args.output_dir = "bert_qat"
    training_args.seed = 42
    training_args.overwrite_output_dir = True
    training_args.eval_steps = 100
    training_args.save_steps = 100
    training_args.greater_is_better = True
    training_args.load_best_model_at_end = True
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    training_args.save_total_limit = 1
    training_args.metric_for_best_model = "accuracy"

    bert_dataset = BertDataset("Intel/bert-base-uncased-mrpc")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=bert_dataset.get_train_dataset(),
        eval_dataset=bert_dataset.get_eval_dataset(),
        compute_metrics=compute_metrics,
        tokenizer=bert_dataset.tokenizer,
        data_collator=bert_dataset.data_collator,
    )

    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()

    model_trained = copy.deepcopy(model)
    model_trained.load_state_dict(torch.load("bert_qat/pytorch_model.bin"), strict=False)
    return model_trained


def compute_metrics(p: EvalPrediction):
    metric = load_metric("glue", "mrpc")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


def qat_post_process(output):
    if isinstance(output, (transformers.modeling_outputs.SequenceClassifierOutput, dict)):
        _, preds = torch.max(output["logits"], dim=1)
    else:
        try:
            _, preds = torch.max(output[0], dim=1)
        except Exception:
            _, preds = torch.max(output, dim=1)
    return preds
