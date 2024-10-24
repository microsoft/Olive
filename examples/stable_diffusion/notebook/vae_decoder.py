import torch
from diffusers import AutoencoderKL


def vae_decoder_inputs(batch_size, torch_dtype):
    return {
        "latent_sample": torch.rand((batch_size, 4, 128, 128), dtype=torch_dtype),
        "return_dict": False,
    }


def _dummy_inputs(model=None):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def _model_loader(model_name):
    model = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    model.forward = model.decode
    return model


def _io_config(model):
    return {
        "input_names": ["latent_sample", "return_dict"],
        "output_names": ["sample"],
        "dynamic_axes": {"latent_sample": {"0": "batch", "1": "channels", "2": "height", "3": "width"}},
    }
