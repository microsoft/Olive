# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Example modified from: https://docs.openvino.ai/2023.2/notebooks/225-stable-diffusion-text-to-image-with-output.html
# --------------------------------------------------------------------------
import numpy as np
import openvino as ov
import torch
from diffusers import StableDiffusionPipeline


def get_sd_pipe(model_id):
    return StableDiffusionPipeline.from_pretrained(model_id)


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_load(model_path):
    pipe = get_sd_pipe(model_path)
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    return text_encoder


def get_text_encoder_example_input():
    return torch.ones((1, 77), dtype=torch.long)


def get_text_encoder_input_shape():
    return [
        (1, 77),
    ]


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_load(model_path):
    pipe = get_sd_pipe(model_path)
    unet = pipe.unet
    unet.eval()
    return unet


def get_unet_example_input():
    encoder_hidden_state = torch.ones((2, 77, 768))
    latents_shape = (2, 4, 512 // 8, 512 // 8)
    latents = torch.randn(latents_shape)
    t = torch.from_numpy(np.array(1, dtype=float))
    return (latents, t, encoder_hidden_state)


def get_unet_input_shape():
    dtype_mapping = {torch.float32: ov.Type.f32, torch.float64: ov.Type.f64}

    dummy_inputs = get_unet_example_input()
    input_info = []
    for input_tensor in dummy_inputs:
        shape = ov.PartialShape(tuple(input_tensor.shape))
        element_type = dtype_mapping[input_tensor.dtype]
        input_info.append((shape, element_type))
    return input_info


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_load(model_path):
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            return self.vae.encode(x=image)["latent_dist"].sample()

    pipe = get_sd_pipe(model_path)
    vae = pipe.vae
    vae_encoder = VAEEncoderWrapper(vae)
    vae_encoder.eval()
    return vae_encoder


def get_vae_encoder_example_input():
    return torch.zeros((1, 3, 512, 512))


def get_vae_encoder_input_shape():
    return [((1, 3, 512, 512),)]


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_load(model_path):
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    pipe = get_sd_pipe(model_path)
    vae = pipe.vae
    vae_decoder = VAEDecoderWrapper(vae)
    vae_decoder.eval()
    return vae_decoder


def get_vae_decoder_example_input():
    return torch.zeros((1, 4, 64, 64))


def get_vae_decoder_input_shape():
    return [((1, 4, 64, 64),)]
