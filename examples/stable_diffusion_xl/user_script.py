# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import random

import config
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection

from olive.data.registry import Registry

# ruff: noqa: T201

# Generated data helpers


class BaseDataLoader:
    def __init__(self, total):
        self.data = []
        self.total = total
        self.data_folders = [config.data_dir / f.name for f in os.scandir(config.data_dir) if f.is_dir()]
        self.data_folders.sort()

    def __getitem__(self, idx):
        if idx >= len(self.data) or idx >= self.total:
            raise StopIteration
        print(f"Process data {idx}")
        return self.data[idx]

    def load(self, file):
        self.data.append({key: torch.from_numpy(value) for key, value in np.load(file).items()})

    def finish_load(self):
        if len(self.data) > self.total:
            self.data = random.sample(self.data, self.total)


class UnetGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        for f in self.data_folders:
            i = 0
            while True:
                file = f / f"unet_{i}.npz"
                if not os.path.exists(file):
                    break
                self.load(file)
                i += 1
        self.finish_load()


class TextEncoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        for f in self.data_folders:
            self.load(f / "text_encoder_10.npz")
        self.finish_load()


class TextEncoder2GeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        for f in self.data_folders:
            self.load(f / "text_encoder_20.npz")
        self.finish_load()


class VaeDecoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        for f in self.data_folders:
            self.load(f / "vae_decoder.npz")
        self.finish_load()


class VaeEncoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        for f in self.data_folders:
            self.load(f / "vae_decoder_output.npz")
        self.finish_load()


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype), label


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batch_size, torch_dtype):
    return {
        "input_ids": torch.zeros((batch_size, 77), dtype=torch_dtype),
        "output_hidden_states": True,
    }


def text_encoder_load(model_name):
    return CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)


@Registry.register_dataloader()
def text_encoder_dataloader(dataset, batch_size, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batch_size, torch.int32)


@Registry.register_dataloader()
def text_encoder_quantize_data_loader(dataset, data_num, *args, **kwargs):
    return TextEncoderGeneratedDataLoader(data_num)


# -----------------------------------------------------------------------------
# TEXT ENCODER 2
# -----------------------------------------------------------------------------


def text_encoder_2_inputs(batch_size, torch_dtype):
    return {
        "input_ids": torch.zeros((batch_size, 77), dtype=torch_dtype),
        "output_hidden_states": True,
    }


def text_encoder_2_load(model_name):
    return CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2")


def text_encoder_2_conversion_inputs(model):
    return text_encoder_2_inputs(1, torch.int64)


@Registry.register_dataloader()
def text_encoder_2_dataloader(dataset, batch_size, **kwargs):
    return RandomDataLoader(text_encoder_2_inputs, batch_size, torch.int64)


@Registry.register_dataloader()
def text_encoder_2_quantize_data_loader(dataset, data_num, *args, **kwargs):
    return TextEncoder2GeneratedDataLoader(data_num)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batch_size, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((2 * batch_size, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((1,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((2 * batch_size, 77, config.cross_attention_dim), dtype=torch_dtype),
    }

    if is_conversion_inputs:
        inputs["additional_inputs"] = {
            "added_cond_kwargs": {
                "text_embeds": torch.rand((2 * batch_size, 1280), dtype=torch_dtype),
                "time_ids": torch.rand((2 * batch_size, config.time_ids_size), dtype=torch_dtype),
            }
        }
    else:
        inputs["text_embeds"] = torch.rand((2 * batch_size, 1280), dtype=torch_dtype)
        inputs["time_ids"] = torch.rand((2 * batch_size, config.time_ids_size), dtype=torch_dtype)

    return inputs


def unet_load(model_name):
    return UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")


def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32, True).values())


@Registry.register_dataloader()
def unet_data_loader(dataset, batch_size, **kwargs):
    return RandomDataLoader(unet_inputs, batch_size, torch.float16)


@Registry.register_dataloader()
def unet_quantize_data_loader(dataset, data_num, *args, **kwargs):
    return UnetGeneratedDataLoader(data_num)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batch_size, torch_dtype):
    return {
        "sample": torch.rand((batch_size, 3, config.vae_sample_size, config.vae_sample_size), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_encoder_load(model_name):
    subfolder = None if model_name == "madebyollin/sdxl-vae-fp16-fix" else "vae"
    model = AutoencoderKL.from_pretrained(model_name, subfolder=subfolder)
    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model


def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def vae_encoder_dataloader(dataset, batch_size, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batch_size, torch.float16)


@Registry.register_dataloader()
def vae_encoder_quantize_data_loader(dataset, data_num, *args, **kwargs):
    return VaeEncoderGeneratedDataLoader(data_num)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batch_size, torch_dtype):
    return {
        "latent_sample": torch.rand(
            (batch_size, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype
        ),
        "return_dict": False,
    }


def vae_decoder_load(model_name):
    subfolder = None if model_name == "madebyollin/sdxl-vae-fp16-fix" else "vae"
    model = AutoencoderKL.from_pretrained(model_name, subfolder=subfolder)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def vae_decoder_dataloader(dataset, batch_size, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batch_size, torch.float16)


@Registry.register_dataloader()
def vae_decoder_quantize_data_loader(dataset, data_num, *args, **kwargs):
    return VaeDecoderGeneratedDataLoader(data_num)
