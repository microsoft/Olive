# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import config
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextModelWithProjection

from olive.data.registry import Registry


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
