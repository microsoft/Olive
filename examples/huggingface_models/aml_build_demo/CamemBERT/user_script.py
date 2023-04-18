# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import evaluate
import numpy
import torch
from datasets import load_dataset
from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import AutoTokenizer, CamembertForTokenClassification

from olive.constants import Framework

# https://huggingface.co/Jean-Baptiste/camembert-ner
model_name = "Jean-Baptiste/camembert-ner"
dataset_name = "Jean-Baptiste/wikiner_fr"
split = "test"


class CalibrationDataLoader(CalibrationDataReader):
    def __init__(self, dataloader, post_func, num_samplers=100):
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.post_func = post_func
        self.counter = 0
        self.num_samplers = num_samplers

    def get_next(self):
        if self.counter >= self.num_samplers:
            return None
        self.counter += 1
        if self.iter is None:
            self.iter = iter(self.dataloader)
        try:
            return self.post_func(next(self.iter))
        except StopIteration:
            return None

    def rewind(self):
        self.iter = None
        self.counter = 0


# -------------------- model -------------------
def load_model(model_path=None):
    model = CamembertForTokenClassification.from_pretrained(model_name)
    model = model.to("cpu")
    return model


# -------------------- dataset -------------------
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = 0 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(0)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        is_split_into_words=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = torch.LongTensor(new_labels)
    return tokenized_inputs


def create_evaluation_dataset():
    dataset = load_dataset(dataset_name, split=split)
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    class _Dateset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            return self.dataset[index], self.dataset[index]["labels"]

        def __len__(self):
            return len(self.dataset)

    return _Dateset(tokenized_datasets)


def create_dataloader(data_dir="", batch_size=2):
    def _collate_fn(batch):
        batch = default_collate(batch)
        return batch

    dataset = create_evaluation_dataset()
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)


def create_cali_dataloader():
    def _post_func(sampler):
        return sampler

    dataloader = create_dataloader()
    cali_dataloader = CalibrationDataLoader(create_dataloader(dataloader, _post_func))
    return cali_dataloader


# -------------------- post process -------------------
def _convert_idx_to_ner_tags(labels):
    id2label = {0: "O", 1: "I-LOC", 2: "I-PER", 3: "I-MISC", 4: "I-ORG"}
    return [id2label[t.item()] for t in labels]


def post_process(model_output, model):
    if model.framework == Framework.ONNX:
        logits = model_output[0]
    else:
        logits = model_output.logits
    predicted_token_class_ids = logits.argmax(-1)
    predicted_tokens_classes = _convert_idx_to_ner_tags(predicted_token_class_ids[0])
    return predicted_tokens_classes


# -------------------- evaluations -------------------
def _evaluate(pre, ref, computer_func=None):
    if computer_func is None:
        return None
    return computer_func.compute(predictions=pre, references=ref)


def evaluate_accuracy_gpu(model, data_dir, batch_size, device="gpu"):
    evaluate_accuracy(model, data_dir, batch_size, device=device)


def evaluate_accuracy(model, data_dir, batch_size, device):
    prepared_model = model.prepare_session(inference_settings=None, device=device)
    dataloader = create_dataloader(batch_size=batch_size)
    seqeval = evaluate.load("seqeval")

    pre = []
    ref = []

    for item in tqdm(dataloader):
        for v in item[-1]:
            ref.append(_convert_idx_to_ner_tags(v))

        item = item[0]
        if model.framework == Framework.ONNX:
            input_ids = numpy.ascontiguousarray(item["input_ids"].cpu().numpy())
            attention_mask = numpy.ascontiguousarray(item["attention_mask"].cpu().numpy())
            input = {"input_ids": input_ids, "attention_mask": attention_mask}
            ort_outputs = prepared_model.run(None, input)
            outputs = post_process(ort_outputs, model)
            pre.append(outputs)

        elif model.framework == Framework.PYTORCH:
            with torch.no_grad():
                ort_outputs = prepared_model(input_ids=item["input_ids"], attention_mask=item["attention_mask"])
                outputs = post_process(ort_outputs, model)
                pre.append(outputs)
    _rls = _evaluate(pre, ref, seqeval)
    rls = _rls["overall_accuracy"]
    return rls
