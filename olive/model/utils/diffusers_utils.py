from pathlib import Path


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
