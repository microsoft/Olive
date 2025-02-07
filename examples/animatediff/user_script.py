from huggingface_hub import model_info
from diffusers import AutoencoderKL, UNet2DConditionModel, UNetMotionModel, MotionAdapter
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch

unet_sample_size=64
cross_attention_dim=768

def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model") or model_name

def unet_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{2}step_diffusers.safetensors"
    base = "stable-diffusion-v1-5/stable-diffusion-v1-5"#"emilianJR/epiCRealism"  # Choose to your favorite base model."stabilityai/stable-diffusion-2-1-base"#

    adapter = MotionAdapter()
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt)))
    model = UNetMotionModel.from_unet2d(model, adapter)
    return model

def unet_inputs(batch_size, torch_dtype, is_conversion_inputs=False):
    num_frames = 16
    # TODO(jstoecker): Rename onnx::Concat_4 to text_embeds and onnx::Shape_5 to time_ids
    inputs = {
        "sample": torch.rand((batch_size, 4, num_frames, unet_sample_size, unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batch_size,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batch_size * num_frames, 77, cross_attention_dim), dtype=torch_dtype),
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

def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32, True).values())