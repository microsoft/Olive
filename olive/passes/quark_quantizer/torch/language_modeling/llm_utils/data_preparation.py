#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


from __future__ import annotations

import logging
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, default_data_collator

logger = logging.getLogger(__name__)


def get_pileval(
    tokenizer: PreTrainedTokenizer, nsamples: int, seqlen: int, device: str | None, seed: int = 0
) -> DataLoader[torch.Tensor]:
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)

    samples = []
    n_run = 0
    for data in dataset:
        line_encoded = tokenizer.encode(data["text"].strip())
        if 0 < len(line_encoded) <= seqlen:
            sample = torch.tensor([line_encoded], device=device)
            samples.append(sample)
            n_run += 1
        if n_run == nsamples:
            break

    # Concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen

    # Create training dataset by splitting concatenated samples
    train_dataset = [cat_samples[:, i * seqlen : (i + 1) * seqlen] for i in range(n_split)]

    # Create batched samples
    return torch.cat(train_dataset, dim=0)


def get_wikitext2(
    tokenizer: PreTrainedTokenizer, nsamples: int, seqlen: int, device: str | None, seed: int = 0
) -> DataLoader[torch.Tensor]:
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    trainenc = trainenc.to(device)

    import random

    random.seed(seed)
    torch.random.manual_seed(seed)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return torch.cat([sample["input_ids"] for sample in traindataset], dim=0)


def get_calib_dataloader_for_benchmark(
    dataset_name: str = "pileval_for_awq_benchmark",
    tokenizer: AutoTokenizer | None = None,
    batch_size: int = 1,
    num_calib_data: int = 128,
    seqlen: int = 2048,
    device: str = "cpu",
) -> DataLoader[torch.Tensor]:
    if dataset_name == "pileval_for_awq_benchmark":
        samples = get_pileval(tokenizer, num_calib_data, seqlen, device, seed=42)
        if batch_size != len(samples):
            logger.info(
                "[INFO-Warning] For AWQ benchmark, batch_size should be %d. Changing batch_size to %d.",
                len(samples),
                len(samples),
            )
            batch_size = len(samples)
    elif dataset_name == "wikitext_for_gptq_benchmark":
        samples = get_wikitext2(tokenizer, num_calib_data, seqlen, device)
    else:
        raise NotImplementedError

    calib_dataloader: DataLoader[list[dict[str, torch.Tensor]]] = DataLoader(
        samples, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return calib_dataloader


def get_calib_dataloader_to_tensor(
    dataset_name: str = "cnn_dailymail",
    tokenizer: AutoTokenizer | None = None,
    batch_size: int = 1,
    num_calib_data: int = 512,
    seqlen: int = 512,
    shuffle: bool = False,
    device: str | None = None,
) -> DataLoader[torch.Tensor]:
    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        text_data = dataset["text"][:num_calib_data]
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_data = dataset["article"][:num_calib_data]
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_data = dataset["text"][:num_calib_data]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=seqlen)
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    return DataLoader(batch_encoded, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def get_calib_dataloader_to_dict(
    dataset_name: str = "cnn_dailymail",
    tokenizer: AutoTokenizer | None = None,
    batch_size: int = 1,
    num_calib_data: int = 512,
    seqlen: int = 512,
    device: str | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    def make_data_block(
        examples: dict[str, list[str]],
        tokenizer: AutoTokenizer | None = None,
        prompt_col_name: str = "",
        max_length: int = 512,
    ) -> dict[str, list[list[torch.Tensor]]]:
        res: dict[str, list[list[torch.Tensor]]] = tokenizer(
            examples[prompt_col_name], padding=True, truncation=True, max_length=max_length
        )
        return res

    def my_collate_fn(blocks: list[dict[str, list[list[str]]]]) -> dict[str, torch.Tensor]:
        data_batch = {}
        data_batch["input_ids"] = torch.Tensor([block["input_ids"] for block in blocks])
        if device:
            data_batch["input_ids"] = data_batch["input_ids"].to(device)
        return data_batch

    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        prompt_col_name = "text"
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        prompt_col_name = "article"
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        prompt_col_name = "text"
    else:
        raise NotImplementedError

    dataset = dataset.select(
        indices=list(range(min(len(dataset), num_calib_data))),
        keep_in_memory=True,
    )
    tokenized_datasets = dataset.map(
        make_data_block,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        remove_columns=dataset.column_names,
        keep_in_memory=True,
        fn_kwargs={"tokenizer": tokenizer, "prompt_col_name": prompt_col_name, "max_length": seqlen},
    )

    return DataLoader(tokenized_datasets, batch_size=batch_size, collate_fn=my_collate_fn)


def get_ultrachat(
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    tokenizer: AutoTokenizer | None = None,
    batch_size: int = 1,
    num_calib_data: int = 512,
    seqlen: int = 512,
    device: str | None = None,
) -> DataLoader[list[dict[str, torch.Tensor]]]:
    max_sequence_length = seqlen

    ds = load_dataset(dataset_name, split="train_sft")
    ds = ds.shuffle(seed=42).select(range(num_calib_data))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    traindataset = []
    for i in range(len(ds["input_ids"])):
        inp = torch.tensor([ds["input_ids"][i]], device=device)
        attention_mask = torch.tensor([ds["attention_mask"][i]], device=device)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

    calib_dataloader: DataLoader[list[dict[str, torch.Tensor]]] = DataLoader(
        traindataset, batch_size=None, shuffle=False
    )
    return calib_dataloader


def get_calib_dataloader(
    dataset_name: str, processor: AutoProcessor | None = None, **kwargs: Any
) -> DataLoader[torch.Tensor] | DataLoader[list[dict[str, torch.Tensor]]] | DataLoader[dict[str, torch.Tensor]]:
    if dataset_name in ["pileval", "cnn_dailymail", "wikitext"]:
        return get_calib_dataloader_to_tensor(dataset_name, **kwargs)
    elif dataset_name in ["pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark"]:
        return get_calib_dataloader_for_benchmark(dataset_name, **kwargs)
    elif "ultrachat" in dataset_name:
        return get_ultrachat(dataset_name, **kwargs)
    else:
        raise NotImplementedError


class ConcatDataset(Dataset):
    def __init__(self, dataset, max_length=4096):
        self.dataset = dataset
        self.samples = []
        buffer = {"input_ids": [], "attention_mask": [], "labels": []}
        for sample in self.dataset:
            buffer = {k: v + sample[k] for k, v in buffer.items()}
            while len(next(iter(buffer.values()))) > max_length:
                self.samples.append({k: v[:max_length] for k, v in buffer.items()})
                buffer = {k: v[max_length:] for k, v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_trainer_dataset(path, subset, tokenizer, max_train_samples, max_eval_samples, seqlen=1024):
    def tokenize_add_label(sample):
        if path in ["wikitext"]:
            input_text = sample["text"]

        elif path in ["shibing624/AdvertiseGen"]:
            input_text = sample["content"] + sample["summary"]
        else:
            raise ValueError(f"Unsupported dataset path: {path}")

        input_ids = tokenizer.encode(tokenizer.bos_token + input_text + tokenizer.eos_token, add_special_tokens=False)

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": input_ids,
        }

    if path in ["wikitext"]:
        train_dataset = load_dataset(path=path, name="wikitext-2-raw-v1", split=subset, trust_remote_code=True)
    elif path in ["shibing624/AdvertiseGen"]:
        train_dataset = load_dataset(path=path, split=subset, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported path: {path}")
    # Using wikitext as default eval_dataset
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)

    if max_train_samples:
        max_train_samples = min(len(train_dataset), max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        logger.info("select %s from training data to build train dataset ...", max_train_samples)

    if max_eval_samples:
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info("select %s from test data to build eval dataset ...", max_eval_samples)

    train_dataset = train_dataset.map(tokenize_add_label, remove_columns=list(train_dataset.features))
    train_dataset = ConcatDataset(train_dataset, seqlen)

    eval_dataset = eval_dataset.map(tokenize_add_label, remove_columns=list(eval_dataset.features))
    eval_dataset = ConcatDataset(eval_dataset, seqlen)
    return {
        "train_dataset": train_dataset,
        "data_collator": default_data_collator,
        "eval_dataset": eval_dataset,
    }


def get_dataset(path, subset, tokenizer, seqlen):
    if path in ["wikitext"]:
        text = load_dataset(path=path, name="wikitext-2-raw-v1", split=subset)
        strtext = "\n\n".join(text["text"])
    elif path in ["shibing624/AdvertiseGen"]:
        text = load_dataset(path=path, split=subset)
        strtext = "\n\n".join(str(i[0]) + str(i[1]) for i in list(zip(list(text["content"]), list(text["summary"]))))
    else:
        raise ValueError(f"Unsupported dataset path: {path}")
    tokenized_text = tokenizer(strtext, return_tensors="pt")
    tokenized_text_len = tokenized_text.input_ids.shape[1]

    sample = [tokenized_text.input_ids[:, i : i + seqlen] for i in range(0, tokenized_text_len - seqlen - 1, seqlen)]
    sample = torch.dstack(sample).squeeze(0).permute(1, 0)

    return TensorDataset(sample)


def get_loader(path, subset, tokenizer, seqlen=1024, num_batch=-1, batch_size=1, shuffle=False):
    dataset = get_dataset(path, subset, tokenizer, seqlen)
    data_size = len(dataset)

    if num_batch != -1:  # num_batch == -1 using the whole dataset
        sample_size = min(data_size, num_batch * batch_size)
        subset_indices = torch.randperm(data_size)[:sample_size]
        dataset = Subset(dataset, subset_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
