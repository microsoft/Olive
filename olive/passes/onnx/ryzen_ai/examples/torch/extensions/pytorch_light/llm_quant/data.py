#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import random
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

def get_c4_data(data_path, tokenizer, seqlen, seed=0, nsamples=256):
    random.seed(seed)
    valdata = load_from_disk(data_path)

    valenc = []
    samples = int(len(valdata))
    if nsamples == -1:
        nsamples = samples
    for _ in tqdm(range(nsamples)):
        if len(valenc) == nsamples:
            break
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.dstack(valenc).squeeze().permute(1, 0)
    return valenc

def get_wiki2_data(data_path, tokenizer, seqlen, seed=0, nsamples=-1):
    random.seed(seed)
    with open(data_path, encoding="utf-8") as f:
        text = f.read()
    tokenized_text = tokenizer(text, return_tensors='pt')
    tokenized_text_len = tokenized_text.input_ids.shape[1]

    valenc = []
    for i in tqdm(range(0, tokenized_text_len - seqlen - 1, seqlen)):
        if len(valenc) == nsamples:
            break
        valenc.append(tokenized_text.input_ids[:, i : i + seqlen])
    valenc = torch.dstack(valenc).squeeze().permute(1, 0)
    return valenc

def get_wiki2_raw_data(data_path, tokenizer, seqlen, seed=0, nsamples=-1):
    random.seed(seed)
    testdata = load_from_disk(data_path)
    tokenized_text = tokenizer("\n".join(testdata['text']), return_tensors='pt')
    # tokenized_text = tokenizer("\n\n".join(testdata['text']), return_tensors='pt') for GPTQ Released code
    tokenized_text_len = tokenized_text.input_ids.shape[1]

    valenc = []
    for i in tqdm(range(0, tokenized_text_len - seqlen - 1, seqlen)):
        if len(valenc) == nsamples:
            break
        valenc.append(tokenized_text.input_ids[:, i : i + seqlen])

    valenc = torch.dstack(valenc).squeeze().permute(1, 0)
    return valenc

def loader(data, batch_size=1, shuffle=False):
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_dataloader(pretrained_model_path, args):
    mapping = {"c4": "c4/validation", "wikitext2": "wikitext2/wiki.test.tokens", "wikitext2-raw": "wikitext_2_raw_v1/test"}
    data_path = args.dataset_path + mapping[args.dataset]
    # data_path = "/group/dphi_algo_scratch_08/zijunx/language_model_dataset/" + mapping[args.dataset]
    mapping = {"c4": get_c4_data, "wikitext2": get_wiki2_data, "wikitext2-raw": get_wiki2_raw_data}
    load_data = mapping[args.dataset]
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, legacy=False, use_fast=False)
    except Exception as ex:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=True)

    valloader = loader(load_data(data_path, tokenizer, args.seqlen))
    optloader = loader(load_data(data_path, tokenizer, args.seqlen, seed = args.seed, nsamples = args.q_opt_samples))
    args.q_opt_samples = len(optloader)
    return optloader, valloader
