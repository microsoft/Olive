#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
import torch
from torch.utils.data import DataLoader
import logging
from typing import List, Optional, Dict, Any, Union
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer


def get_pileval(tokenizer: PreTrainedTokenizer,
                nsamples: int,
                seqlen: int,
                device: Optional[str],
                seed: int = 0) -> List[Dict[str, torch.Tensor]]:

    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir='data_cache')
    dataset = dataset.shuffle(seed=seed)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        sample = sample.to(device)
        samples.append(sample)
        n_run += 1
        if n_run == nsamples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    logging.debug(f" * Split into {n_split} blocks")
    traindataset = []
    for i in range(n_split):
        traindataset.append({"input_ids": cat_samples[:, i * seqlen:(i + 1) * seqlen]})
    return traindataset


def get_wikitext2(tokenizer: PreTrainedTokenizer,
                  nsamples: int,
                  seqlen: int,
                  device: Optional[str],
                  seed: int = 0) -> List[Dict[str, torch.Tensor]]:

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir='data_cache')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
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
        traindataset.append({'input_ids': inp, 'attention_mask': attention_mask})
    return traindataset


def get_calib_dataloader_to_list(dataset_name: str = "pileval_for_awq_benchmark",
                                 tokenizer: AutoTokenizer = None,
                                 batch_size: int = 1,
                                 num_calib_data: int = 128,
                                 seqlen: int = 2048,
                                 device: str = 'cpu') -> DataLoader[List[Dict[str, torch.Tensor]]]:
    if dataset_name == "pileval_for_awq_benchmark":
        samples = get_pileval(tokenizer, num_calib_data, seqlen, device, seed=42)
    elif dataset_name == "wikitext_for_gptq_benchmark":
        samples = get_wikitext2(tokenizer, num_calib_data, seqlen, device)
    else:
        raise NotImplementedError

    calib_dataloader: DataLoader[List[Dict[str, torch.Tensor]]] = DataLoader(samples, batch_size=None,
                                                                             shuffle=False)  # type: ignore

    return calib_dataloader


def get_calib_dataloader_to_tensor(dataset_name: str = "cnn_dailymail",
                                   tokenizer: AutoTokenizer = None,
                                   batch_size: int = 1,
                                   num_calib_data: int = 512,
                                   seqlen: int = 512,
                                   device: Optional[str] = None) -> DataLoader[torch.Tensor]:
    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir='data_cache')
        text_data = dataset["text"][:num_calib_data]
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train", cache_dir='data_cache')
        text_data = dataset["article"][:num_calib_data]
    elif dataset_name == "wikitext":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir='data_cache')
        text_data = dataset["text"][:num_calib_data]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=seqlen)
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)

    return calib_dataloader


def get_calib_dataloader_to_dict(dataset_name: str = "cnn_dailymail",
                                 tokenizer: AutoTokenizer = None,
                                 batch_size: int = 1,
                                 num_calib_data: int = 512,
                                 seqlen: int = 512,
                                 device: Optional[str] = None) -> DataLoader[Dict[str, torch.Tensor]]:

    def make_data_block(examples: Dict[str, List[str]],
                        tokenizer: AutoTokenizer = None,
                        prompt_col_name: str = '',
                        max_length: int = 512) -> dict[str, List[List[torch.Tensor]]]:
        res: dict[str, List[List[torch.Tensor]]] = tokenizer(examples[prompt_col_name],
                                                             padding=True,
                                                             truncation=True,
                                                             max_length=max_length)
        return res

    def my_collate_fn(blocks: List[Dict[str, List[List[str]]]]) -> Dict[str, torch.Tensor]:
        data_batch = {}
        data_batch["input_ids"] = torch.Tensor([block["input_ids"] for block in blocks])
        if device:
            data_batch["input_ids"] = data_batch["input_ids"].to(device)
        return data_batch

    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation", cache_dir='data_cache')
        prompt_col_name = "text"
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train", cache_dir='data_cache')
        prompt_col_name = "article"
    elif dataset_name == "wikitext":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir='data_cache')
        prompt_col_name = "text"
    else:
        raise NotImplementedError

    dataset = dataset.select(
        indices=[i for i in range(min(len(dataset), num_calib_data))],
        keep_in_memory=True,
    )
    tokenized_datasets = dataset.map(make_data_block,
                                     batched=True,
                                     batch_size=len(dataset),
                                     num_proc=1,
                                     remove_columns=dataset.column_names,
                                     keep_in_memory=True,
                                     fn_kwargs={
                                         'tokenizer': tokenizer,
                                         'prompt_col_name': prompt_col_name,
                                         'max_length': seqlen
                                     })

    calib_dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, collate_fn=my_collate_fn)

    return calib_dataloader


def get_calib_dataloader(
    dataset_name: str, **kwargs: Any
) -> Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]]]:
    if dataset_name in ["pileval", "cnn_dailymail"]:
        return get_calib_dataloader_to_tensor(dataset_name, **kwargs)
    elif dataset_name in ["pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark"]:
        return get_calib_dataloader_to_list(dataset_name, **kwargs)
    else:
        raise NotImplementedError
