# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Optional

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


# Available preprocessing steps (executed in this order)
AVAILABLE_STEPS = [
    "image_filtering",
    "auto_caption",
    "auto_tagging",
    "image_resizing",
    "aspect_ratio_bucketing",
]

# Default steps if user doesn't specify
DEFAULT_STEPS = {"aspect_ratio_bucketing": {}}


@Registry.register_pre_process("image_lora_preprocess")
def image_lora_preprocess(
    dataset,
    base_resolution: int = 512,
    steps: Optional[dict[str, dict]] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    overwrite: bool = False,
):
    """Preprocessing chain for Stable Diffusion LoRA/DreamBooth training.

    Steps are executed in order: image_filtering -> auto_caption -> auto_tagging
    -> image_resizing -> aspect_ratio_bucketing

    If `steps` is not provided, defaults to {"aspect_ratio_bucketing": {}}.
    If `steps` is provided, only those steps are executed (no defaults).

    Args:
        dataset: The dataset to process.
        base_resolution: Base resolution (512 for SD1.5, 1024 for SDXL/Flux).
        steps: Steps to run with their parameters. If not provided, uses default.
            Example: {"image_filtering": {"min_size": 256}, "auto_caption": {"model_type": "florence2"}}
        output_dir: Output directory for processed images.
        device: Device for neural network operations.
        overwrite: Whether to overwrite existing files.

    Returns:
        Processed dataset.

    Example:
        # LoRA (default: just bucketing)
        image_lora_preprocess(dataset)

        # LoRA with auto-captioning
        image_lora_preprocess(dataset, steps={
            "auto_caption": {"model_type": "blip2"},
            "aspect_ratio_bucketing": {}
        })

        # Fixed size (no bucketing)
        image_lora_preprocess(dataset, steps={"image_resizing": {}})

    """
    import inspect

    # Use default steps if not provided
    if steps is None:
        steps = DEFAULT_STEPS.copy()

    # Run steps in order
    for step_name in AVAILABLE_STEPS:
        if step_name not in steps:
            continue

        # Get step parameters
        params = steps[step_name].copy()

        # Add default params based on step type
        if step_name == "aspect_ratio_bucketing" and "base_resolution" not in params:
            params["base_resolution"] = base_resolution
        if step_name == "image_resizing" and "target_resolution" not in params:
            params["target_resolution"] = base_resolution

        logger.info("Running preprocess step: %s", step_name)
        try:
            preprocess_fn = Registry.get_pre_process_component(step_name)

            # Get function signature to filter global params
            sig = inspect.signature(preprocess_fn)
            fn_params = set(sig.parameters.keys())
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

            # Add global params if function accepts them
            global_params = {"device": device, "overwrite": overwrite, "output_dir": output_dir}
            for k, v in global_params.items():
                if k not in params and v is not None and (accepts_kwargs or k in fn_params):
                    params[k] = v

            dataset = preprocess_fn(dataset, **params)
        except KeyError:
            logger.warning("Preprocess component '%s' not found, skipping", step_name)
        except Exception:
            logger.exception("Error in preprocess step '%s'", step_name)
            raise

    return dataset
