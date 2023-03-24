# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy

import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from datasets.utils.logging import disable_progress_bar
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

disable_progress_bar()


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
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
        return result

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
    model_trained.load_state_dict(torch.load("{}/pytorch_model.bin".format("bert_qat")), strict=False)
    return model_trained


def compute_metrics(p: EvalPrediction):
    metric = load_metric("glue", "mrpc")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


def create_benchmark_dataloader(data_dir, batchsize):
    bert_dataset = BertDataset("Intel/bert-base-uncased-mrpc")
    eval_dataloader = torch.utils.data.DataLoader(
        BertDatasetWrapper(bert_dataset.get_eval_dataset()), batch_size=batchsize, drop_last=True
    )
    return eval_dataloader


def post_process(output):
    if isinstance(output, transformers.modeling_outputs.SequenceClassifierOutput) or isinstance(output, dict):
        _, preds = torch.max(output["logits"], dim=1)
    else:
        try:
            _, preds = torch.max(output[0], dim=1)
        except Exception:
            _, preds = torch.max(output, dim=1)
    return preds


def load_pytorch_origin_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained("Intel/bert-base-uncased-mrpc")
    model.eval()
    return model
