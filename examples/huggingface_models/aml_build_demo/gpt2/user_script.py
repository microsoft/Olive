# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnxruntime as ort
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

ort.set_default_logger_severity(3)

# https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
model_name = "gpt2"
max_length = 50
min_length = 1
num_beams = 1
num_return_sequences = 1
length_penalty = 1
repetition_penalty = 1


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to("cpu")
    return model


# -------------------- dataset -------------------
def create_dataloader(data_dir=None, batch_size=1):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    sentences = ["The product is released"] * batch_size
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

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
