import json
from pathlib import Path
from typing import List

import mteb
import numpy as np
import torch
from transformers import AutoTokenizer

from olive.constants import Framework
from olive.engine.footprint import Footprint, FootprintNode
from olive.model import OliveModelHandler
from olive.workflows import run as olive_run


class OliveEncoder:
    def __init__(self, model, session):
        self.model = model
        self.session = session
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.total = 0
        self.max_len = 0

    def encode(self, corpus: List, **kwargs):
        if self.model.framework == Framework.ONNX:
            encoded_input = self.tokenizer(
                corpus, padding="max_length", max_length=128, truncation=True, return_tensors="np"
            )
            # batch_size is 1 for static model
            model_outputs = []
            for i in range(len(corpus)):
                model_inputs = {
                    "input_ids": encoded_input.input_ids[i : i + 1, :].astype(np.int64),
                    "attention_mask": encoded_input.attention_mask[i : i + 1, :].astype(np.int64),
                    "token_type_ids": encoded_input.token_type_ids[i : i + 1, :].astype(np.int64),
                }
                model_output = self.model.run_session(self.session, model_inputs)[0]
                model_outputs.append(model_output[0])
            model_output = np.array(model_outputs)
        elif self.model.framework == Framework.PYTORCH:
            encoded_input = self.tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
            model_inputs = {
                "input_ids": encoded_input.input_ids,
                "attention_mask": encoded_input.attention_mask,
                "token_type_ids": encoded_input.token_type_ids,
            }
            self.max_len = max(self.max_len, model_inputs["input_ids"].shape[1])
            print(self.max_len)
            with torch.no_grad():
                model_output = self.model.run_session(self.session, model_inputs)
            model_output = model_output.last_hidden_state.numpy()
        # select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.
        model_output = model_output[:, 0, :]
        self.total += len(corpus)
        print(self.total)
        return model_output


def eval_accuracy(model: OliveModelHandler, device, execution_providers, tasks):
    sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)

    evaluation = mteb.MTEB(tasks=tasks)
    oliveEncoder = OliveEncoder(model, sess)
    results = evaluation.run(oliveEncoder, output_folder=None)
    return results[0].scores["test"][0]["main_score"]


if __name__ == "__main__":
    # Greedy search for the best combination of ops to quantize
    all_ops = [
        "Mul",
        "Transpose",
        "Unsqueeze",
        "Add",
        "Softmax",
        "Gelu",
        "LayerNormalization",
        "Gather",
        "MatMul",
        "Sub",
        "Where",
        "Expand",
        "Gemm",
        "Tanh",
        "Reshape",
    ]
    target_accuracy = 0.8
    with Path("bge-small-en-v1.5.json").open() as fin:
        olive_config = json.load(fin)
    for i, op in enumerate(all_ops):
        if op in olive_config["passes"]["OnnxQuantization"]["op_types_to_quantize"]:
            continue
        olive_config["passes"]["OnnxQuantization"]["op_types_to_quantize"].append(op)
        result = olive_run(olive_config)
        footprint: Footprint = next(iter(result.values()))
        node: FootprintNode = next(iter(footprint.nodes.values()))
        accuracy = node.metrics.value["accuracy-accuracy_custom"].value
        print(f"Ops: {olive_config["passes"]["OnnxQuantization"]["op_types_to_quantize"]} Accuracy: {accuracy}")
        if accuracy < target_accuracy:
            olive_config["passes"]["OnnxQuantization"]["op_types_to_quantize"].remove(op)
    print(f"Used Ops: {olive_config["passes"]["OnnxQuantization"]["op_types_to_quantize"]}")
