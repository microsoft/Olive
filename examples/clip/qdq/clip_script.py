from __future__ import annotations

from collections import OrderedDict
from itertools import chain

import torch
from transformers import (
    AutoProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry

HF_MODEL_SUBFOLDER_MAPPING = {
    "sentence-transformers/clip-ViT-B-32": "0_CLIPModel",
}


def load_image_encoder(model_name):
    return CLIPVisionModelWithProjection.from_pretrained(
        model_name,
        subfolder=HF_MODEL_SUBFOLDER_MAPPING.get(model_name, ""),
    ).eval()


def load_text_encoder(model_name):
    return CLIPTextModelWithProjection.from_pretrained(
        model_name,
        subfolder=HF_MODEL_SUBFOLDER_MAPPING.get(model_name, ""),
    ).eval()


def hfdataset_pre_process_for_clip(
    dataset,
    processor,
    torch_model=None,
    image_col: str | None = None,
    caption_col: str | None = None,
    label_col: str = "label",
    max_samples: int | None = None,
    max_length: int = 77,
    batch_size: int = 32,
):
    def generate_inputs(sample, indices):
        captions = sample.get(caption_col, None)
        images = sample.get(image_col, None)

        kwargs = {
            "padding": "max_length",
            "max_length": max_length,
            "truncation": True,
            "add_special_tokens": True,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = [img.convert("RGB") for img in images]
        if captions:
            kwargs["text"] = list(chain([x[0] for x in captions]))

        encoded_input = processor(**kwargs)

        return {
            **encoded_input,
            label_col: torch_model(**encoded_input)[0] if torch_model else sample.get(label_col, indices),
        }

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    tokenized_datasets = dataset.map(
        generate_inputs,
        batched=True,
        batch_size=batch_size,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )
    tokenized_datasets.set_format("torch", output_all_columns=True)

    return tokenized_datasets


@Registry.register_pre_process()
def pre_process_dataset(
    dataset,
    model_name: str,
    generate_ground_truth: bool = False,
    image_col: str | None = None,
    caption_col: str | None = None,
    label_col: str = "label",
    max_samples: int | None = None,
    max_length: int = 77,
    **kwargs,
):
    if image_col is None and caption_col is None:
        raise ValueError("Either image_col or caption_col must be provided.")

    if generate_ground_truth:
        if image_col and caption_col:
            raise ValueError("Can not generate two types of embedding at the same time.")

        torch_model = load_image_encoder(model_name) if image_col else load_text_encoder(model_name)
    else:
        torch_model = None

    processor = AutoProcessor.from_pretrained(model_name)
    dataset = hfdataset_pre_process_for_clip(
        dataset,
        processor,
        torch_model=torch_model,
        image_col=image_col,
        caption_col=caption_col,
        label_col=label_col,
        max_length=max_length,
        max_samples=max_samples,
    )
    return BaseDataset(dataset, label_col)


@Registry.register_post_process()
def embed_post_process(output):
    """Post-processing for CLIP output."""
    if isinstance(output, (dict, OrderedDict)):
        if "embeds" in output:
            return output["embeds"]
        elif "text_embeds" in output:
            return output["text_embeds"]
        elif "image_embeds" in output:
            return output["image_embeds"]
    elif isinstance(output, torch.Tensor):
        return output.argmax(dim=-1)
    raise ValueError(f"Unsupported output type: {type(output)}")


def eval_similarity_degrad(output, targets, batch_size=1024):
    import torch.nn.functional as F

    preds = output.preds
    scores = [
        # pylint: disable=E1102
        F.cosine_similarity(preds[i : i + batch_size], targets[i : i + batch_size])
        # pylint: enable=E1102
        for i in range(0, preds.size(0), batch_size)
    ]
    return {"percentage": f"{100.0 - torch.mean(torch.cat(scores)) * 100.0:.2f}"}
