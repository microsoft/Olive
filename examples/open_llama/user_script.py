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
            pass


def calib_dataloader(data_dir, batch_size, *args, **kwargs):
    model_path = kwargs.pop("model_path")
    return PileDataloader(model_path, batch_size=batch_size)


def eval_accuracy(model: OliveModelHandler, device, execution_providers, batch_size):
    from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import LMEvalParser, evaluate

    results = {}
    if model.framework == Framework.PYTORCH:
        eval_args = LMEvalParser(
            model="hf",
            model_args=f"pretrained={model.model_path},tokenizer={model_id},dtype=float32",
            batch_size=batch_size,
            tasks="lambada_openai",
            device="cpu",
        )
        results = evaluate(eval_args)

    elif model.framework == Framework.ONNX:
        output_config_file = Path(model.model_path).resolve().parent / "config.json"
        config.to_json_file(output_config_file, use_diff=False)
        eval_args = LMEvalParser(
            model="hf",
            model_args=f"pretrained={Path(model.model_path).resolve().parent},tokenizer={model_id},model_format=onnx",
            batch_size=batch_size,
            tasks="lambada_openai",
            device="cpu",
        )
        results = evaluate(eval_args)
    return results["results"]["lambada_openai"]["acc,none"]
