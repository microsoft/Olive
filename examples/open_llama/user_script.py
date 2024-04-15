# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from transformers import AutoConfig, LlamaTokenizer

from olive.constants import Framework
from olive.model import OliveModelHandler

model_id = "openlm-research/open_llama_3b"
config = AutoConfig.from_pretrained(model_id)


class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype, model_framework=Framework.PYTORCH):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.model_framework = model_framework

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype, self.model_framework), label


def dummy_inputs(batch_size, torch_dtype, model_framework=Framework.PYTORCH):
    past_sequence_length = 1
    attention_mask_sequence_length = 1
    sequence_length = 2

    inputs = {
        "input_ids": torch.randint(10, (batch_size, sequence_length), dtype=torch.int64),
        "attention_mask": torch.randint(10, (batch_size, attention_mask_sequence_length), dtype=torch.int64),
    }
    rand_kv_tensor = torch.rand(
        (
            batch_size,
            config.num_attention_heads,
            past_sequence_length,
            int(config.hidden_size / config.num_attention_heads),
        ),
        dtype=torch_dtype,
    )
    if model_framework == Framework.ONNX:
        for layer_index in range(config.num_hidden_layers):
            inputs[f"past_key_values.{layer_index}.key"] = rand_kv_tensor
            inputs[f"past_key_values.{layer_index}.value"] = rand_kv_tensor
        inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)
    elif model_framework == Framework.PYTORCH:
        inputs["use_cache"] = True
        inputs["past_key_values"] = [torch.stack((rand_kv_tensor, rand_kv_tensor))] * config.num_hidden_layers
    return inputs


def dataloader_func(data_dir, batch_size, *args, **kwargs):
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    return RandomDataLoader(dummy_inputs, batch_size, torch.float16, model_framework)


def tokenize_function(examples):
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    return tokenizer(examples["text"])


class PileDataloader:
    def __init__(self, model_path, batch_size=1, seqlen=2048, sub_folder="train"):
        random.seed(0)
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.dataset = load_dataset("NeelNanda/pile-10k", split=sub_folder)
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        self.sess = None
        model_path = Path(model_path).resolve()
        if model_path.parent.stem == "decoder_with_past_model":
            decoder_model_path = None
            for item in Path(model_path).parent.parent.glob("decoder_model/*.onnx"):
                decoder_model_path = item.resolve()
                break
            self.sess = ort.InferenceSession(decoder_model_path, providers=["CPUExecutionProvider"])

    def __iter__(self):
        try:
            while True:
                while True:
                    i = random.randint(0, len(self.dataset) - 1)
                    trainenc = self.dataset[i]
                    if trainenc["input_ids"].shape[0] > self.seqlen:
                        break
                i = random.randint(0, trainenc["input_ids"].shape[0] - self.seqlen - 1)
                j = i + self.seqlen
                inp = trainenc["input_ids"][i:j].unsqueeze(0)
                mask = torch.ones(inp.shape)
                if self.sess is None:
                    yield {
                        "input_ids": inp.detach().cpu().numpy().astype("int64"),
                        "attention_mask": mask.detach().cpu().numpy().astype("int64"),
                    }, 0
                else:
                    outputs = self.sess.run(
                        None,
                        {
                            "input_ids": inp[:, :-1].detach().cpu().numpy().astype("int64"),
                            "attention_mask": mask[:, :-1].detach().cpu().numpy().astype("int64"),
                        },
                    )
                    ort_input = {}
                    ort_input["input_ids"] = inp[:, -1].unsqueeze(0).detach().cpu().numpy().astype("int64")
                    for layer_index in range(config.num_hidden_layers):
                        ort_input[f"past_key_values.{layer_index}.key"] = outputs[layer_index * 2 + 1]
                        ort_input[f"past_key_values.{layer_index}.value"] = outputs[layer_index * 2 + 2]

                    ort_input["attention_mask"] = np.zeros(
                        [self.batch_size, ort_input["past_key_values.0.key"].shape[2] + 1], dtype="int64"
                    )
                    yield ort_input, 0

        except StopIteration:
            return


def calib_dataloader(data_dir, batch_size, *args, **kwargs):
    model_path = kwargs.pop("model_path")
    return PileDataloader(model_path, batch_size=batch_size)


def eval_accuracy(model: OliveModelHandler, data_dir, batch_size, device, execution_providers):
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate

    results = {}
    if model.framework == Framework.PYTORCH:
        results = evaluate(
            model="hf-causal",
            model_args=(
                f"pretrained={model.model_path or model.hf_config.model_name},tokenizer={model_id},dtype=float32"
            ),
            batch_size=batch_size,
            tasks=["lambada_openai"],
        )
    elif model.framework == Framework.ONNX:
        output_config_file = Path(model.model_path).resolve().parent / "config.json"
        config.to_json_file(output_config_file, use_diff=False)
        results = evaluate(
            model="hf-causal",
            model_args=f"pretrained={Path(model.model_path).resolve().parent},tokenizer={model_id}",
            batch_size=batch_size,
            tasks=["lambada_openai"],
            model_format="onnx",
        )
    return results["results"]["lambada_openai"]["acc"]
