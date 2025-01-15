# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import model_info
from sd_utils import config
from transformers.models.clip.modeling_clip import CLIPTextModel
import numpy as np
from olive.data.registry import Registry
import os

# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype, total = 1):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.total = total

    def __getitem__(self, idx):
        print("getitem: " + str(idx))
        if idx >= self.total: return None
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype) #, label

class BaseDataLoader:
    def __init__(self):
        self.data = []

    def __getitem__(self, idx):
        print("getitem: " + str(idx))
        if idx >= len(self.data): return None
        return self.data[idx]

folders = [f"data/{f.name}" for f in os.scandir('data') if f.is_dir()]
folders = folders[:1]
print(folders)

def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model") or model_name

# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batch_size, torch_dtype):
    return {"tokens": torch.zeros((batch_size, 77), dtype=torch_dtype)}


def text_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    print('text_encoder_load: ' + base_model_id)
    model = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    return model


def text_encoder_conversion_inputs(model=None):
    return text_encoder_inputs(1, torch.int32)


@Registry.register_dataloader()
def text_encoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batch_size, torch.int32)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batch_size, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "latent": torch.rand((batch_size, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "time_emb": torch.rand((batch_size,), dtype=torch_dtype),
        "text_emb": torch.rand((batch_size, 77, config.cross_attention_dim), dtype=torch_dtype),
    }
    return inputs


def get_unet_ov_example_input():
    import numpy as np

    encoder_hidden_state = torch.ones((2, 77, 768))
    latents_shape = (2, 4, 512 // 8, 512 // 8)
    latents = torch.randn(latents_shape)
    t = torch.from_numpy(np.array(1, dtype=float))
    return (latents, t, encoder_hidden_state)


def unet_load(model_name):
    base_model_id = get_base_model_name(model_name)
    print("base_model_id: " + base_model_id)
    # variant="fp16" also produces a model > 2GB
    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())

class UnetDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        for f in folders:
            for i in range(5):
                latent = torch.from_numpy(np.fromfile(f + f'/{i}_latent.raw', dtype=np.float32).reshape(1, 4, 64, 64))
                time = torch.from_numpy(np.fromfile(f + f'/{i}_time.raw', dtype=np.float32).reshape(1))
                text = torch.from_numpy(np.fromfile(f + f'/{i}_text.raw', dtype=np.float32).reshape(1, 77, 1024))
                self.data.append({ "latent": latent, "time_emb": time, "text_emb": text })
                text = torch.from_numpy(np.fromfile(f + f'/{i}_untext.raw', dtype=np.float32).reshape(1, 77, 1024))
                self.data.append({ "latent": latent, "time_emb": time, "text_emb": text })

@Registry.register_dataloader()
def unet_data_loader(dataset, batch_size, *args, **kwargs):
    return UnetDataLoader()
    return RandomDataLoader(unet_inputs, batch_size, torch.float32)

# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batch_size, torch_dtype):
    data = torch.rand((batch_size, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype)
    data = (data * 2 - 1) / 0.18215
    return {
        "latent": data 
    }

def vae_decoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())

class DecoderDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()
        for f in folders:
            data = torch.from_numpy(np.fromfile(f + '/latent.raw', dtype=np.float32).reshape(1, 4, 64, 64))
            self.data.append({ "latent": data })


@Registry.register_dataloader()
def vae_decoder_data_loader(dataset, batch_size, *args, **kwargs):
    return DecoderDataLoader()
    return RandomDataLoader(vae_decoder_inputs, batch_size, torch.float32, 100)

