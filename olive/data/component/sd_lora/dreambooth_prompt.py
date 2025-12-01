# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""DreamBooth prompt generation component."""

import logging
import random
from pathlib import Path
from typing import Optional

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_pre_process()
def dreambooth_prompt(
    dataset,
    prompt_template: str = "a photo of {identifier} {classname}",
    class_prompt_template: str = "a photo of {classname}",
    identifier_token: str = "sks",
    class_token: str = "dog",
    prompt_variations: Optional[list[str]] = None,
    use_filewords: bool = False,
    caption_extension: str = ".txt",
    overwrite: bool = False,
    **kwargs,
):
    """Generate DreamBooth-style prompts for images.

    DreamBooth uses fixed prompt templates with a unique identifier token
    to teach the model a new concept.

    Common templates:
    - "a photo of {identifier} {classname}"
    - "a {identifier} {classname}"
    - "a photo of a {identifier} {classname}"
    - "{identifier} {classname}, high quality"

    Args:
        dataset: The dataset to process.
        prompt_template: Template for instance prompts. Use {identifier} and {classname}.
        class_prompt_template: Template for class prompts. Use {classname}.
        identifier_token: Unique identifier (e.g., "sks", "xyz", "ohwx").
        class_token: Class noun (e.g., "dog", "woman", "art style").
        prompt_variations: List of additional prompt variations.
        use_filewords: Append words from filename to prompt.
        caption_extension: Extension for saving prompt files.
        overwrite: Overwrite existing prompt files.
        **kwargs: Additional arguments (ignored).

    Returns:
        Dataset with prompts generated.
    """
    # Build base prompts
    instance_prompt = prompt_template.format(identifier=identifier_token, classname=class_token)
    class_prompt = class_prompt_template.format(classname=class_token)

    # Build prompt variations
    all_instance_prompts = [instance_prompt]
    if prompt_variations:
        for variation in prompt_variations:
            all_instance_prompts.append(variation.format(identifier=identifier_token, classname=class_token))

    processed_count = 0
    skipped_count = 0

    for i in range(len(dataset)):
        item = dataset[i]
        image_path = Path(item["image_path"])
        caption_path = image_path.with_suffix(caption_extension)

        if not overwrite and caption_path.exists():
            skipped_count += 1
            continue

        # Select prompt
        if len(all_instance_prompts) > 1:
            prompt = random.choice(all_instance_prompts)
        else:
            prompt = instance_prompt

        # Optionally append filewords
        if use_filewords:
            filename = image_path.stem
            filewords = filename.replace("_", " ").replace("-", " ")
            skip_words = {"img", "image", "photo", "pic", "picture", str(i)}
            words = [w for w in filewords.split() if w.lower() not in skip_words and not w.isdigit()]
            if words:
                prompt = f"{prompt}, {' '.join(words)}"

        # Save prompt
        caption_path.write_text(prompt, encoding="utf-8")
        processed_count += 1

    logger.info("Generated %d DreamBooth prompts, skipped %d existing", processed_count, skipped_count)

    # Store prompts in dataset for reference
    dataset.instance_prompt = instance_prompt
    dataset.class_prompt = class_prompt
    dataset.identifier_token = identifier_token
    dataset.class_token = class_token

    return dataset


@Registry.register_pre_process()
def generate_class_images(
    dataset,
    class_data_dir: str,
    class_prompt: str = "a photo of dog",
    num_class_images: int = 200,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    batch_size: int = 4,
    device: str = "cuda",
    use_fp16: bool = True,
    seed: Optional[int] = None,
    skip_if_exists: bool = True,
    **kwargs,
):
    """Generate class/regularization images for prior preservation.

    Prior preservation helps prevent language drift by training on both
    instance images and generated class images.

    Args:
        dataset: The dataset (unchanged, but class images are generated).
        class_data_dir: Directory to save generated class images.
        class_prompt: Prompt for generating class images.
        num_class_images: Number of class images to generate.
        model_name: Stable Diffusion model to use.
        guidance_scale: CFG scale for generation.
        num_inference_steps: Number of denoising steps.
        batch_size: Batch size for generation.
        device: Device to run on.
        use_fp16: Use FP16 precision.
        seed: Random seed.
        skip_if_exists: Skip if enough images exist.
        **kwargs: Additional arguments.

    Returns:
        Dataset (unchanged).
    """
    import torch
    from diffusers import StableDiffusionPipeline

    class_path = Path(class_data_dir)
    class_path.mkdir(parents=True, exist_ok=True)

    # Check existing images
    existing_images = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
    if skip_if_exists and len(existing_images) >= num_class_images:
        logger.info("Found %d existing class images, skipping generation", len(existing_images))
        return dataset

    images_to_generate = num_class_images - len(existing_images)
    if images_to_generate <= 0:
        return dataset

    logger.info("Generating %d class images with prompt: '%s'", images_to_generate, class_prompt)

    # Load pipeline
    dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Generate images
    generated_count = len(existing_images)
    for batch_start in range(0, images_to_generate, batch_size):
        batch_count = min(batch_size, images_to_generate - batch_start)

        with torch.no_grad():
            images = pipe(
                prompt=[class_prompt] * batch_count,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images

        for img in images:
            save_path = class_path / f"class_{generated_count:05d}.png"
            img.save(save_path)
            generated_count += 1

        logger.info("Generated %d / %d class images", generated_count, num_class_images)

    # Cleanup
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    logger.info("Class image generation complete: %d images in %s", generated_count, class_path)

    return dataset
