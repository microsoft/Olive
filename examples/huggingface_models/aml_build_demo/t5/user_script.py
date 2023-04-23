# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import evaluate
import onnxruntime as ort
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from olive.constants import Framework

ort.set_default_logger_severity(3)

# https://huggingface.co/t5-small
model_name = "t5-small"
dataset_name = "cnn_dailymail"
subset = "1.0.0"
split = "validation"

max_length = 1000
min_length = 1
num_beams = 4
num_return_sequences = 1
length_penalty = 1
repetition_penalty = 1


def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to("cpu")
    return model


# -------------------- dataset -------------------
def tokenize_and_align_labels(examples):
    data = [f"summarize: {example}" for example in examples["article"]]
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    tokenizer.padding_side = "left"
    tokenized_inputs = tokenizer(
        data,
        truncation=True,
        padding=True,
        return_tensors="pt",
        # max_length=128,
        # return_overflowing_tokens=True,
    )
    # pre process
    tokenized_inputs["labels"] = examples["highlights"]
    return tokenized_inputs


def create_evaluation_dataset():
    raw_dataset = load_dataset(dataset_name, subset, split=split)
    tokenized_datasets = raw_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset.column_names,
    )
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"], output_all_columns=True)

    class _Dateset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            return self.dataset[index], self.dataset[index]["labels"]

        def __len__(self):
            return 5
            # return len(self.dataset)

    return _Dateset(tokenized_datasets)


def create_dataloader(data_dir="", batch_size=2):
    def _collate_fn(batch):
        batch = default_collate(batch)
        batch[0].update(
            {
                "max_length": torch.IntTensor([max_length]),
                "min_length": torch.IntTensor([min_length]),
                "num_beams": torch.IntTensor([num_beams]),
                "num_return_sequences": torch.IntTensor([num_return_sequences]),
                "length_penalty": torch.FloatTensor([length_penalty]),
                "repetition_penalty": torch.FloatTensor([repetition_penalty]),
            }
        )
        batch[0]["input_ids"] = batch[0]["input_ids"].to(torch.int32)
        return batch

    dataset = create_evaluation_dataset()
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)


def _evaluate(pre, ref, computer_func=None):
    if computer_func is None:
        return None
    return computer_func.compute(predictions=pre, references=ref)


def evaluate_accuracy_gpu(model, data_dir, batch_size, device="gpu"):
    evaluate_accuracy(model, data_dir, batch_size, device=device)


def post_process(model_output):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    (batch_size, num_sequences, _) = model_output.shape
    ort_decoded_sequences = []
    for i in range(batch_size):
        for j in range(num_sequences):
            decoded_sequence = tokenizer.decode(model_output[i][j], skip_special_tokens=True)
            ort_decoded_sequences.append(decoded_sequence)
    return ort_decoded_sequences


def evaluate_accuracy(model, data_dir, batch_size, device):
    prepared_model = model.prepare_session(inference_settings=None, device=device)
    dataloader = create_dataloader(batch_size=batch_size)
    rough = evaluate.load("rouge")

    pre = []
    ref = []

    for item in tqdm(dataloader):
        for label in item[-1]:
            ref.append(label)
        item = item[0]
        if model.framework == Framework.ONNX:
            input = {k: v.cpu().numpy() for k, v in item.items() if k != "labels" and k != "attention_mask"}
            ort_outputs = prepared_model.run(None, input)[0]
            outputs = post_process(ort_outputs)
            pre.extend(outputs)
    _rls = _evaluate(pre, ref, rough)
    rls = _rls["rouge1"]
    return rls


def _generate(model, item):
    model.generate(
        input_ids=item["input_ids"],
        decoder_start_token_id=model.config.decoder_start_token_id,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        min_length=min_length,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
    )


def evaluate_torch_latency(model, data_dir, batch_size, device):
    import time

    prepared_model = model.prepare_session(inference_settings=None, device=device)
    dataloader = create_dataloader(batch_size=batch_size)
    latency = []
    for item in dataloader:
        item = item[0]
        if device == "gpu":
            prepared_model = prepared_model.to("cuda")
            item = {k: v.to("cuda") for k, v in item.items() if k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            for _ in range(10):
                _generate(prepared_model, item)
        with torch.no_grad():
            for _ in range(20):
                t = time.perf_counter()
                _generate(prepared_model, item)
                latency.append(time.perf_counter() - t)
        break
    latency_metrics = {
        "latency": round(sum(latency) / len(latency) * 1000, 5),
    }
    print("latency_metrics: ", latency_metrics)
    return latency_metrics
