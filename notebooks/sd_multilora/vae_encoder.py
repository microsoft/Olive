import torch
from diffusers import AutoencoderKL


def vae_encoder_inputs(batch_size, torch_dtype):
    return {
        "sample": torch.rand((batch_size, 3, 1024, 1024), dtype=torch_dtype),
        "return_dict": False,
    }


def _dummy_inputs(model=None):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def _model_loader(model_name):
    model = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model


def _io_config(model):
    return {
        "input_names": ["latent_sample", "return_dict"],
        "output_names": ["sample"],
        "dynamic_axes": {
            "latent_sample": {"0": "batch_size", "1": "num_channels_latent", "2": "height_latent", "3": "width_latent"},
            "sample": {"0": "batch_size", "1": "num_channels", "2": "height", "3": "width"},
        },
    }
