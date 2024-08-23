# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def vision_embed_tokens_loader(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return model.model.vision_embed_tokens


def get_dummy_inputs(model=None):
    processor = AutoProcessor.from_pretrained(model.model_path, trust_remote_code=True)
    user_prompt = "<|user|>\n"
    assistant_prompt = "<|assistant|>\n"
    prompt_suffix = "<|end|>\n"
    prompt = f"{user_prompt}<|image_1|>\nWhat is shown in this image?{prompt_suffix}{assistant_prompt}"
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True, timeout=10).raw)
    inputs = processor(prompt, image, return_tensors="pt")
    return (
        inputs["pixel_values"],
        inputs["image_sizes"],
    )


def text_embedding_loader(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    return model.model.embed_tokens
