from __future__ import annotations

from itertools import chain
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPProcessor


def get_ort_ep_policy(p):
    from onnxruntime import OrtExecutionProviderDevicePolicy as Policy

    mapping = {
        "DEFAULT": Policy.DEFAULT,
        "PREFER_CPU": Policy.PREFER_CPU,
        "PREFER_GPU": Policy.PREFER_GPU,
        "PREFER_NPU": Policy.PREFER_NPU,
        "MAX_PERFORMANCE": Policy.MAX_PERFORMANCE,
        "MAX_EFFICIENCY": Policy.MAX_EFFICIENCY,
        "MIN_OVERALL_POWER": Policy.MIN_OVERALL_POWER,
    }
    return mapping.get(p)


class OrtModule:
    def __init__(self, model_path: Path, policy="PREFER_NPU", **kwargs) -> None:
        self.model_path = model_path
        self.session = self._init_ort_session(policy, **kwargs)

        self._input_names = [i.name for i in self.session.get_inputs()]
        self._outputs_names = [o.name for o in self.session.get_outputs()]
        self._batch_size = self.session.get_inputs()[0].shape[0]
        self._latency_trace = []

    def _init_ort_session(self, policy, **kwargs):
        policy = get_ort_ep_policy(policy)
        if policy is None:
            raise ValueError(f"Invalid EP selection policy: {policy}")

        sess_options = ort.SessionOptions()
        sess_options.set_provider_selection_policy(policy)

        return ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
        )

    def run(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = {
            name: tensor.split(self._batch_size, dim=0) for name, tensor in tensors.items() if name in self._input_names
        }
        missing_inputs = self._input_names - inputs.keys()
        if missing_inputs:
            raise RuntimeError(f"Missing inputs for ONNX model: {missing_inputs}")

        # Split batches and convert torch tensors to numpy arrays
        batches = [dict(zip(inputs.keys(), [v.numpy() for v in values])) for values in zip(*inputs.values())]
        # Run the ONNX model
        start = perf_counter()
        outputs = [self.session.run(None, batch) for batch in tqdm(batches)]
        self._latency_trace.append((len(batches), perf_counter() - start))

        return dict(
            zip(
                self._outputs_names,
                [torch.from_numpy(np.concatenate(a)) for a in zip(*outputs)],
            )
        )

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def latency(self):
        latencies = np.concatenate(
            [np.full(num_batches, total_time / num_batches * 1000) for num_batches, total_time in self._latency_trace]
        )
        return (
            round(np.mean(latencies).item(), 2),
            round(np.percentile(latencies, 90).item(), 2),
        )


class OrtCLIPModel:
    def __init__(
        self,
        model_name: str,
        text_model_path: Path,
        vision_model_path: Path,
        tokenizer_name: str | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.text_model = OrtModule(text_model_path)
        self.vision_model = OrtModule(vision_model_path)

    def get_image_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        output = self.vision_model.run(inputs)
        return output["embeds"]

    def get_text_features(self, text):
        max_length = self.text_model.session.get_inputs()[0].shape[1]
        inputs = self.tokenizer(
            text=text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        output = self.text_model.run(
            {
                "input_ids": inputs.input_ids.int(),
                "attention_mask": inputs.attention_mask.int(),
                # "attention_mask": self._create_4d_mask(
                #     inputs["attention_mask"],
                #     inputs["input_ids"].shape,
                # ),
            }
        )
        return output["embeds"]

    def _create_4d_mask(self, mask, input_shape, masked_value=-50.0):
        # (batch_size, num_heads, seq_len, head_dim)
        batch_sz, seq_len = input_shape
        expanded_mask = mask[:, None, None, :].expand(batch_sz, 1, seq_len, seq_len)
        inverted_mask = 1.0 - expanded_mask.float()
        return inverted_mask.masked_fill(inverted_mask.bool(), masked_value)


def compute_logits(text_embeds, image_embeds, logit_scale=100.0):
    text_embeds = np.asarray(text_embeds)
    image_embeds = np.asarray(image_embeds)

    text_norm = text_embeds / (np.linalg.norm(text_embeds, axis=1, keepdims=True) + 1e-8)
    image_norm = image_embeds / (np.linalg.norm(image_embeds, axis=1, keepdims=True) + 1e-8)
    return np.dot(text_norm, image_norm.T) * logit_scale


def compute_topk_accuracy(logits, labels, k=1):
    topk_preds = np.argsort(logits, axis=-1)[:, -k:]  # Get top-k indices
    labels = np.array(labels).reshape(-1, 1)
    correct = (topk_preds == labels).any(axis=1)
    return {f"top{k}_accuracy": correct.mean()}


def eval_retrieval_accuracy(
    model_name,
    text_model_path,
    vision_model_path,
    dataset_name,
    split,
    tokenizer_name=None,
):
    dataset = load_dataset(dataset_name, split=split)
    stacked_captions = dataset["caption"]
    captions = list(chain(*stacked_captions))
    images = dataset["image"]

    model = OrtCLIPModel(
        model_name,
        text_model_path=text_model_path,
        vision_model_path=vision_model_path,
        tokenizer_name=tokenizer_name,
    )
    text_embeds = model.get_text_features(captions)
    image_embeds = model.get_image_features(images)
    logits = compute_logits(text_embeds, image_embeds)

    labels = list(
        chain(
            *[
                [i] * len(c)
                for i, c in zip(
                    range(len(stacked_captions)),
                    stacked_captions,
                )
            ]
        )
    )

    print("Text encoding latency", model.text_model.latency)
    print("Image encoding latency", model.vision_model.latency)

    return {
        **compute_topk_accuracy(logits, labels, k=1),
        **compute_topk_accuracy(logits, labels, k=5),
    }


def format_output(data, ratio=100.0):
    import json

    formatted = json.loads(json.dumps(data), parse_float=lambda x: round(float(x) * ratio, 2))
    return json.dumps(formatted, indent=2)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a NPU model")
    parser.add_argument(
        "--model",
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--text-encoder",
        "--text-encoder-path",
        type=Path,
        default="models/openai/clip_b32/text/model.onnx",
    )
    parser.add_argument(
        "--image-encoder",
        "--image-encoder-path",
        type=Path,
        default="models/openai/clip_b32/image/model.onnx",
    )
    parser.add_argument(
        "--tokenizer",
        "--tokenizer-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        type=str,
        default="nlphuji/flickr_1k_test_image_text_retrieval",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    result = eval_retrieval_accuracy(
        args.model,
        args.text_encoder,
        args.image_encoder,
        args.dataset,
        args.split,
        args.tokenizer,
    )

    print("Evaluation result", format_output(result))
