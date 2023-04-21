# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnxruntime as ort
import torch
from transformers import GPT2Tokenizer

ort.set_default_logger_severity(3)

# https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
model_name = "gpt2"


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
