from huggingface_hub import model_info
from diffusers import UNet2DConditionModel, UNetMotionModel, MotionAdapter
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

step=2
unet_sample_size=64
cross_attention_dim=768
num_frames = 16

def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model") or model_name

def unet_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"

    adapter = MotionAdapter()
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt)))
    model = UNetMotionModel.from_unet2d(model, adapter)
    return model

def unet_inputs(batch_size, torch_dtype):
    inputs = {
        "sample": torch.rand((batch_size, 4, num_frames, unet_sample_size, unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batch_size,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batch_size * num_frames, 77, cross_attention_dim), dtype=torch_dtype),
    }
    return inputs

def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32).values())