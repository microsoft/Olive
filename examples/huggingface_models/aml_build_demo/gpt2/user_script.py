# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx
import torch
from transformers import GPT2Tokenizer

from olive.common.utils import run_subprocess

# https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
model_name = "gpt2"


# -------------------- model -------------------
def load_model(model_path):
    model_name = "gpt2"
    run_subprocess(
        f"python -m onnxruntime.transformers.convert_generation -m {model_name}"
        + f" --model_type {model_name} --output {model_path}"
    )
    return onnx.load(model_path)


# -------------------- dataset -------------------
def create_dataloader(data_dir=None, batch_size=1):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    sentences = ["The product is released"] * batch_size
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

    max_length = 50
    min_length = 1
    num_beams = 1
    num_return_sequences = 1
    length_penalty = 1
    repetition_penalty = 1

    tensor_inputs = {
        "input_ids": input_ids.to(torch.int32),
        "max_length": torch.IntTensor([max_length]),
        "min_length": torch.IntTensor([min_length]),
        "num_beams": torch.IntTensor([num_beams]),
        "num_return_sequences": torch.IntTensor([num_return_sequences]),
        "length_penalty": torch.FloatTensor([length_penalty]),
        "repetition_penalty": torch.FloatTensor([repetition_penalty]),
    }
    return ((tensor_inputs, 1),)
