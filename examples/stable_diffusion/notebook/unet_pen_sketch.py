import torch
from diffusers import UNet2DConditionModel


def unet_inputs(batch_size, torch_dtype):
    return {
        "sample": torch.rand((2 * batch_size, 4, 128, 128), dtype=torch_dtype),
        "timestep": torch.rand((1,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((2 * batch_size, 77, 2048), dtype=torch_dtype),
        "additional_inputs": {
            "added_cond_kwargs": {
                "text_embeds": torch.rand((2 * batch_size, 1280), dtype=torch_dtype),
                "time_ids": torch.rand((2 * batch_size, 6), dtype=torch_dtype),
            }
        },
    }


def _dummy_inputs(model):
    return tuple(unet_inputs(1, torch.float32).values())


def _model_loader(model_name):
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    model.load_attn_procs("lora-library/B-LoRA-pen_sketch", weight_name="pytorch_lora_weights.safetensors")
    return model


def _io_config(model):
    return {
        "input_names": ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        "output_names": ["out_sample"],
        "dynamic_axes": {
            "sample": {
                "0": "unet_sample_batch",
                "1": "unet_sample_channels",
                "2": "unet_sample_height",
                "3": "unet_sample_width",
            },
            "timestep": {"0": "unet_time_batch"},
            "encoder_hidden_states": {"0": "unet_hidden_batch", "1": "unet_hidden_sequence"},
            "text_embeds": {"0": "unet_text_embeds_batch", "1": "unet_text_embeds_size"},
            "time_ids": {"0": "unet_time_ids_batch", "1": "unet_time_ids_size"},
        },
    }
