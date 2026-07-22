# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Processor-ready AI2D calibration data for Qwen2.5-VL multimodal quantization."""

# ruff: noqa: INP001

from __future__ import annotations

from collections import Counter

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor

from olive.data.registry import Registry

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _encode_calibration_example(
    processor,
    image,
    question: str,
    answer: str,
    max_image_size: int,
):
    """Encode one supervised image question with expanded decoder masks."""
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    full_messages = [
        *user_messages,
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]

    prompt_text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    image = image.convert("RGB")
    image.thumbnail((max_image_size, max_image_size))
    prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")
    encoded = processor(text=[full_text], images=[image], return_tensors="pt")

    prompt_length = prompt["input_ids"].shape[-1]
    if not torch.equal(prompt["input_ids"], encoded["input_ids"][:, :prompt_length]):
        raise ValueError("The answer-bearing Qwen2.5-VL sequence does not preserve the prompt token prefix.")
    if "mm_token_type_ids" not in encoded:
        raise ValueError("Qwen2.5-VL processor output is missing mm_token_type_ids.")

    labels = torch.full_like(encoded["input_ids"], -100)
    labels[:, prompt_length:] = encoded["input_ids"][:, prompt_length:]
    vision_mask = encoded["mm_token_type_ids"].eq(1)
    answer_mask = labels.ne(-100)
    if not vision_mask.any() or not answer_mask.any():
        raise ValueError("Calibration examples must contain both expanded vision tokens and answer tokens.")

    model_inputs = dict(encoded)
    model_inputs.update(
        {
            "labels": labels,
            "vision_mask": vision_mask,
            "answer_mask": answer_mask,
        }
    )
    return model_inputs, labels


class Qwen25VLAI2DCalibrationDataset(Dataset):
    """Lazily encode AI2D examples with decoder-coordinate MBQ masks."""

    def __init__(self, dataset, model_path: str, max_image_size: int):
        self.dataset = dataset
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.max_image_size = max_image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        choices = "\n".join(f"{_LETTERS[i]}. {choice}" for i, choice in enumerate(row["options"]))
        question = f"{row['question']}\n{choices}\nAnswer with the option letter only."
        answer = _LETTERS[int(row["answer"])]
        return _encode_calibration_example(
            self.processor,
            row["image"],
            question,
            answer,
            self.max_image_size,
        )


class Qwen25VLTextVQACalibrationDataset(Dataset):
    """Lazily encode TextVQA training examples for held-out benchmark calibration."""

    def __init__(self, dataset, model_path: str, max_image_size: int):
        self.dataset = dataset
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.max_image_size = max_image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        answers = [answer.strip() for answer in row["answers"] if answer.strip()]
        if not answers:
            raise ValueError("TextVQA calibration examples must contain a non-empty answer.")
        answer = Counter(answers).most_common(1)[0][0]
        question = (
            f"{row['question']}\nAnswer the question using a short phrase. Do not provide reasoning or additional text."
        )
        return _encode_calibration_example(
            self.processor,
            row["image"],
            question,
            answer,
            self.max_image_size,
        )


@Registry.register_dataset("qwen25vl_ai2d_calibration_dataset")
def qwen25vl_ai2d_calibration_dataset(
    data_name: str = "lmms-lab/ai2d",
    max_samples: int = 1,
):
    """Load a deterministic AI2D calibration prefix."""
    dataset = load_dataset(data_name, split="test", download_mode="reuse_dataset_if_exists")
    return dataset.select(range(min(max_samples, len(dataset))))


@Registry.register_pre_process("qwen25vl_ai2d_calibration_preprocess")
def qwen25vl_ai2d_calibration_preprocess(
    dataset,
    model_path: str,
    max_image_size: int = 448,
):
    """Create a lazy processor-ready dataset for the Olive calibration loader."""
    return Qwen25VLAI2DCalibrationDataset(dataset, model_path, max_image_size)


@Registry.register_dataset("qwen25vl_textvqa_train_calibration_dataset")
def qwen25vl_textvqa_train_calibration_dataset(
    data_name: str = "lmms-lab/textvqa",
    max_samples: int = 16,
):
    """Load a deterministic TextVQA training prefix for held-out calibration."""
    dataset = load_dataset(data_name, split="train", download_mode="reuse_dataset_if_exists")
    candidate_count = min(max_samples * 4, len(dataset))
    candidates = dataset.select(range(candidate_count))
    valid_indices = [
        index for index, answers in enumerate(candidates["answers"]) if any(answer.strip() for answer in answers)
    ][:max_samples]
    if len(valid_indices) < max_samples:
        raise ValueError(
            f"Only found {len(valid_indices)} valid TextVQA answers in the first {candidate_count} training examples."
        )
    return candidates.select(valid_indices)


@Registry.register_pre_process("qwen25vl_textvqa_calibration_preprocess")
def qwen25vl_textvqa_calibration_preprocess(
    dataset,
    model_path: str,
    max_image_size: int = 448,
):
    """Create lazy TextVQA calibration batches for Qwen2.5-VL."""
    return Qwen25VLTextVQACalibrationDataset(dataset, model_path, max_image_size)
