import copy
from pathlib import Path


def get_vae_encoder(vae):
    """Create a VAE encoder model for export by patching forward method."""
    vae_encoder = copy.deepcopy(vae)
    vae_encoder.forward = lambda sample: vae_encoder.encode(sample).latent_dist.parameters
    return vae_encoder


def get_vae_decoder(vae):
    """Create a VAE decoder model for export by patching forward method."""
    vae_decoder = copy.deepcopy(vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(latent_sample).sample
    return vae_decoder


def is_valid_diffusers_model(model_path: str) -> bool:
    """Check if the path is a valid diffusion model.

    Diffusion models are identified by the presence of a model_index.json file.

    Args:
        model_path: Local path or HuggingFace model ID.

    Returns:
        True if the path points to a valid diffusion model.

    """
    # Local path
    path = Path(model_path)
    if path.is_dir():
        return (path / "model_index.json").exists()

    # HuggingFace model ID - try to check if model_index.json exists
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(model_path, "model_index.json")
        return True
    except Exception:
        return False
