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
from olive.data.registry import Registry
import os
import numpy as np
import sys

# Generated data helpers

class BaseDataLoader:
    def __init__(self, total):
        self.data = []
        self.total = total
        self.data_folders = [config.data_dir / f.name for f in os.scandir(config.data_dir) if f.is_dir()]

    def __getitem__(self, idx):
        print("getitem: " + str(idx))
        if idx >= len(self.data) or idx >= self.total: return None
        return self.data[idx]

class UnetGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        latent_min = sys.float_info.max
        latent_max = sys.float_info.min
        time_min = sys.float_info.max
        time_max = sys.float_info.min
        text_min = sys.float_info.max
        text_max = sys.float_info.min
        
        for f in self.data_folders:
            text = torch.from_numpy(np.fromfile(f + f'/text_embeds.raw', dtype=np.float32).reshape(1, 77, 1024))
            text_max = max(text_max, text.max())
            text_min = min(text_min, text.min())
            text_neg = torch.from_numpy(np.fromfile(f + f'/neg_embeds.raw', dtype=np.float32).reshape(1, 77, 1024))
            text_max = max(text_max, text_neg.max())
            text_min = min(text_min, text_neg.min())
            for i in range(10000):
                if os.path.exists(f + f'/{i}_latent.raw') == False: break

                latent = torch.from_numpy(np.fromfile(f + f'/{i}_latent.raw', dtype=np.float32).reshape(1, 4, 64, 64))
                latent_max = max(latent_max, latent.max())
                latent_min = min(latent_min, latent.min())
                time = torch.from_numpy(np.fromfile(f + f'/{i}_timestep.raw', dtype=np.float32).reshape(1))
                time_max = max(time_max, time.max())
                time_min = min(time_min, time.min())
                self.data.append({ "sample": latent, "timestep": time, "encoder_hidden_states": text })
                self.data.append({ "sample": latent, "timestep": time, "encoder_hidden_states": text_neg })
        print(latent_min, latent_max, time_min, time_max, text_min, text_max)


class TextEncoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        latent_min = sys.float_info.max
        latent_max = sys.float_info.min
        for f in self.data_folders:
            data = torch.from_numpy(np.fromfile(f + '/text_inputs.raw', dtype=np.int32).reshape(1, 77))
            latent_max = max(latent_max, data.max())
            latent_min = min(latent_min, data.min())
            self.data.append({ "input_ids": data })
            data = torch.from_numpy(np.fromfile(f + '/uncond_input.raw', dtype=np.int32).reshape(1, 77))
            latent_max = max(latent_max, data.max())
            latent_min = min(latent_min, data.min())
            self.data.append({ "input_ids": data })
        print(latent_min, latent_max)


class VaeDecoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        latent_min = sys.float_info.max
        latent_max = sys.float_info.min
        for f in self.data_folders:
            data = torch.from_numpy(np.fromfile(f + '/latent.raw', dtype=np.float32).reshape(1, 4, 64, 64))
            latent_max = max(latent_max, data.max())
            latent_min = min(latent_min, data.min())
            self.data.append({ "latent_sample": data })
        print(latent_min, latent_max)


class VaeEncoderGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        latent_min = sys.float_info.max
        latent_max = sys.float_info.min
        for f in self.data_folders:
            data = torch.from_numpy(np.fromfile(f + '/output_img.raw', dtype=np.float32).reshape(1, 3, 512, 512))
            latent_max = max(latent_max, data.max())
            latent_min = min(latent_min, data.min())
            self.data.append({ "sample": data })
        print(latent_min, latent_max)

# TODO clean this up

def get_data_list(size, torch_dtype, total, value_min, value_max):
    result = []
    result.append(torch.zeros(size, dtype=torch_dtype))
    result.append(torch.zeros(size, dtype=torch_dtype) + value_min)
    result.append(torch.zeros(size, dtype=torch_dtype) + value_max)
    total -= 3
    if total <= 0: return result
    for i in range(total):
        result.append(torch.rand(size, dtype=torch_dtype) * (value_max - value_min) + value_min)
    return result


class UnetDataRandomLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        samples = get_data_list((1, 4, config.unet_sample_size, config.unet_sample_size), torch.float32, total, -10.9, 7.96)
        timesteps = get_data_list((1), torch.float32, total, 0, 1000)
        states = get_data_list((1, 77, config.cross_attention_dim), torch.float32, total, -7.16, 13.1)
        for i in range(self.total):
            self.data.append({ "sample": samples[i], "timestep": timesteps[i], "encoder_hidden_states": states[i] })


class TextEncoderDataRandomLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        samples = get_data_list((1, 77), torch.float32, total, 0, 49407)
        for i in range(self.total):
            self.data.append({ "input_ids": samples[i].to(torch.int32) })


class VaeDecoderDataRandomLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        samples = get_data_list((1, 4, config.unet_sample_size, config.unet_sample_size), torch.float32, total, -61.3, 49.4)
        for i in range(self.total):
            self.data.append({ "latent_sample": samples[i] })


class VaeEncoderDataRandomLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        samples = get_data_list((1, 3, config.vae_sample_size, config.vae_sample_size), torch.float32, total, -1, 1)
        for i in range(self.total):
            self.data.append({ "sample": samples[i] })


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype), label


def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model") or model_name


def is_lora_model(model_name):
    # TODO(jstoecker): might be a better way to detect (e.g. presence of LORA weights file)
    return model_name != get_base_model_name(model_name)


# Merges LoRA weights into the layers of a base model
def merge_lora_weights(base_model, lora_model_id, submodel_name="unet", scale=1.0):
    import inspect
    from collections import defaultdict
    from functools import reduce

    try:
        from diffusers.loaders import LORA_WEIGHT_NAME
    except ImportError:
        # moved in version 0.24.0
        from diffusers.loaders.lora import LORA_WEIGHT_NAME
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from diffusers.utils.hub_utils import _get_model_file

    parameters = inspect.signature(_get_model_file).parameters

    kwargs = {}
    if "use_auth_token" in parameters:
        kwargs["use_auth_token"] = None
    elif "token" in parameters:
        kwargs["token"] = None

    # Load LoRA weights
    model_file = _get_model_file(
        lora_model_id,
        weights_name=LORA_WEIGHT_NAME,
        cache_dir=None,
        force_download=False,
        resume_download=False,
        proxies=None,
        local_files_only=False,
        revision=None,
        subfolder=None,
        user_agent={
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        },
        **kwargs,
    )
    lora_state_dict = torch.load(model_file, map_location="cpu")

    # All keys in the LoRA state dictionary should have 'lora' somewhere in the string.
    keys = list(lora_state_dict.keys())
    assert all("lora" in k for k in keys)

    if all(key.startswith(submodel_name) for key in keys):
        # New format (https://github.com/huggingface/diffusers/pull/2918) supports LoRA weights in both the
        # unet and text encoder where keys are prefixed with 'unet' or 'text_encoder', respectively.
        submodel_state_dict = {k: v for k, v in lora_state_dict.items() if k.startswith(submodel_name)}
    else:
        # Old format. Keys will not have any prefix. This only applies to unet, so exit early if this is
        # optimizing the text encoder.
        if submodel_name != "unet":
            return
        submodel_state_dict = lora_state_dict

    # Group LoRA weights into attention processors
    attn_processors = {}
    lora_grouped_dict = defaultdict(dict)
    for key, value in submodel_state_dict.items():
        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
        lora_grouped_dict[attn_processor_key][sub_key] = value

    for key, value_dict in lora_grouped_dict.items():
        rank = value_dict["to_k_lora.down.weight"].shape[0]
        cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
        hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

        attn_processors[key] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
        )
        attn_processors[key].load_state_dict(value_dict)

    # Merge LoRA attention processor weights into existing Q/K/V/Out weights
    for name, proc in attn_processors.items():
        attention_name = name[: -len(".processor")]
        attention = reduce(getattr, attention_name.split(sep="."), base_model)
        attention.to_q.weight.data += scale * torch.mm(proc.to_q_lora.up.weight, proc.to_q_lora.down.weight)
        attention.to_k.weight.data += scale * torch.mm(proc.to_k_lora.up.weight, proc.to_k_lora.down.weight)
        attention.to_v.weight.data += scale * torch.mm(proc.to_v_lora.up.weight, proc.to_v_lora.down.weight)
        attention.to_out[0].weight.data += scale * torch.mm(proc.to_out_lora.up.weight, proc.to_out_lora.down.weight)


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batch_size, torch_dtype):
    return torch.zeros((batch_size, 77), dtype=torch_dtype)


def text_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    if is_lora_model(model_name):
        merge_lora_weights(model, model_name, "text_encoder")
    return model


def text_encoder_conversion_inputs(model=None):
    return text_encoder_inputs(1, torch.int32)


@Registry.register_dataloader()
def text_encoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batch_size, torch.int32)


@Registry.register_dataloader()
def text_encoder_quantize_data_loader(dataset, batch_size, *args, **kwargs):
    if config.rand_data:
        return TextEncoderDataRandomLoader(config.data_num)
    return TextEncoderGeneratedDataLoader(config.data_num)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batch_size, torch_dtype, is_conversion_inputs=False):
    # TODO(jstoecker): Rename onnx::Concat_4 to text_embeds and onnx::Shape_5 to time_ids
    inputs = {
        "sample": torch.rand((batch_size, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batch_size,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batch_size, 77, config.cross_attention_dim), dtype=torch_dtype),
    }

    # use as kwargs since they won't be in the correct position if passed along with the tuple of inputs
    kwargs = {
        "return_dict": False,
    }
    if is_conversion_inputs:
        inputs["additional_inputs"] = {
            **kwargs,
            "added_cond_kwargs": {
                "text_embeds": torch.rand((1, 1280), dtype=torch_dtype),
                "time_ids": torch.rand((1, 5), dtype=torch_dtype),
            },
        }
    else:
        inputs.update(kwargs)
        inputs["onnx::Concat_4"] = torch.rand((1, 1280), dtype=torch_dtype)
        inputs["onnx::Shape_5"] = torch.rand((1, 5), dtype=torch_dtype)

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
    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    if is_lora_model(model_name):
        merge_lora_weights(model, model_name, "unet")
    return model


def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())


@Registry.register_dataloader()
def unet_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batch_size, torch.float16)


@Registry.register_dataloader()
def unet_quantize_data_loader(dataset, batch_size, *args, **kwargs):
    if config.rand_data:
        return UnetDataRandomLoader(config.data_num)
    return UnetGeneratedDataLoader(config.data_num)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batch_size, torch_dtype):
    return {"sample": torch.rand((batch_size, 3, config.vae_sample_size, config.vae_sample_size), dtype=torch_dtype)}


def vae_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.forward = lambda sample: model.encode(sample)[0].sample()
    return model


def vae_encoder_conversion_inputs(model=None):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def vae_encoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batch_size, torch.float16)


@Registry.register_dataloader()
def vae_encoder_quantize_data_loader(dataset, batch_size, *args, **kwargs):
    if config.rand_data:
        return VaeEncoderDataRandomLoader(config.data_num)
    return VaeEncoderGeneratedDataLoader(config.data_num)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batch_size, torch_dtype):
    return {
        "latent_sample": torch.rand(
            (batch_size, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype
        )
    }


def vae_decoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def vae_decoder_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batch_size, torch.float16)


@Registry.register_dataloader()
def vae_decoder_quantize_data_loader(dataset, batch_size, *args, **kwargs):
    if config.rand_data:
        return VaeDecoderDataRandomLoader(config.data_num)
    return VaeDecoderGeneratedDataLoader(config.data_num)


# -----------------------------------------------------------------------------
# SAFETY CHECKER
# -----------------------------------------------------------------------------


def safety_checker_inputs(batch_size, torch_dtype):
    return {
        "clip_input": torch.rand((batch_size, 3, 224, 224), dtype=torch_dtype),
        "images": torch.rand((batch_size, config.vae_sample_size, config.vae_sample_size, 3), dtype=torch_dtype),
    }


def safety_checker_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = StableDiffusionSafetyChecker.from_pretrained(base_model_id, subfolder="safety_checker")
    model.forward = model.forward_onnx
    return model


def safety_checker_conversion_inputs(model=None):
    return tuple(safety_checker_inputs(1, torch.float32).values())


@Registry.register_dataloader()
def safety_checker_data_loader(dataset, batch_size, *args, **kwargs):
    return RandomDataLoader(safety_checker_inputs, batch_size, torch.float16)
