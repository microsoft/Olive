# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Auto captioning components for Stable Diffusion LoRA training."""

import logging
from pathlib import Path
from typing import Optional, Union

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


def _load_image(image_path: str):
    """Load an image from path."""
    from PIL import Image

    return Image.open(image_path).convert("RGB")


def _save_caption(
    image_path: str,
    caption: str,
    caption_extension: str = ".txt",
    dataset=None,
    index: Optional[int] = None,
) -> None:
    """Save caption to file or in-memory storage.

    For HuggingFaceImageDataset, captions are stored in memory.
    For other datasets (local folders), captions are saved to text files.

    Args:
        image_path: Path to the image file.
        caption: Caption text to save.
        caption_extension: Extension for caption files (for file-based storage).
        dataset: Dataset object (to check type and store in memory if HuggingFaceImageDataset).
        index: Dataset index (for in-memory storage).
    """
    # Check if dataset supports in-memory caption storage
    if dataset is not None and hasattr(dataset, "set_caption"):
        # Store in memory (for HuggingFaceImageDataset)
        if index is not None:
            dataset.set_caption(index, caption)
        else:
            dataset.set_caption(image_path, caption)
    else:
        # Save to file (for local folder datasets)
        caption_path = Path(image_path).with_suffix(caption_extension)
        caption_path.write_text(caption, encoding="utf-8")


@Registry.register_pre_process()
def auto_caption(
    dataset,
    model_type: str = "blip2",
    model_name: Optional[str] = None,
    prompt: Optional[str] = None,
    max_new_tokens: int = 50,
    batch_size: int = 1,
    device: str = "cuda",
    caption_extension: str = ".txt",  # Default to .txt to match dataset's default
    overwrite: bool = False,
    prefix: str = "",
    suffix: str = "",
    **kwargs,
):
    """Auto-generate captions for images using vision-language models.

    Supports multiple captioning backends:
    - blip2: BLIP-2 model for image captioning
    - florence2: Florence-2 model for detailed image descriptions
    - cogvlm: CogVLM for high-quality captions
    - llava: LLaVA model for conversational image understanding

    Args:
        dataset: The dataset to process.
        model_type: Type of captioning model ("blip2", "florence2", "cogvlm", "llava").
        model_name: Specific model name/path. If None, uses default for model_type.
        prompt: Custom prompt for caption generation.
        max_new_tokens: Maximum number of tokens to generate.
        batch_size: Batch size for processing.
        device: Device to run inference on ("cuda", "cpu").
        caption_extension: Extension for caption files.
        overwrite: Whether to overwrite existing captions.
        prefix: Prefix to add to all generated captions.
        suffix: Suffix to add to all generated captions.
        **kwargs: Additional model-specific arguments.

    Returns:
        The dataset with captions generated and saved.
    """
    if model_type == "blip2":
        return blip2_caption(
            dataset,
            model_name=model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
            caption_extension=caption_extension,
            overwrite=overwrite,
            prefix=prefix,
            suffix=suffix,
            **kwargs,
        )
    elif model_type == "florence2":
        return florence2_caption(
            dataset,
            model_name=model_name,
            task=prompt or "<DETAILED_CAPTION>",
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
            caption_extension=caption_extension,
            overwrite=overwrite,
            prefix=prefix,
            suffix=suffix,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: blip2, florence2")


@Registry.register_pre_process()
def blip2_caption(
    dataset,
    model_name: Optional[str] = None,
    prompt: Optional[str] = None,
    max_new_tokens: int = 50,
    batch_size: int = 1,
    device: str = "cuda",
    caption_extension: str = ".txt",
    overwrite: bool = False,
    prefix: str = "",
    suffix: str = "",
    use_fp16: bool = True,
    **kwargs,
):
    """Generate captions using BLIP-2 model.

    Args:
        dataset: The dataset to process.
        model_name: BLIP-2 model name. Default: "Salesforce/blip2-opt-2.7b".
        prompt: Optional prompt for conditional captioning.
        max_new_tokens: Maximum number of tokens to generate.
        batch_size: Batch size for processing.
        device: Device to run inference on.
        caption_extension: Extension for caption files.
        overwrite: Whether to overwrite existing captions.
        prefix: Prefix to add to all generated captions.
        suffix: Suffix to add to all generated captions.
        use_fp16: Whether to use FP16 for inference.
        **kwargs: Additional generation arguments.

    Returns:
        The dataset with captions generated and saved.
    """
    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    model_name = model_name or "Salesforce/blip2-opt-2.7b"
    logger.info("Loading BLIP-2 model: %s", model_name)

    dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    # Check if dataset supports in-memory caption storage
    use_memory_storage = hasattr(dataset, "set_caption")

    # Process images
    processed_count = 0
    skipped_count = 0

    for i in range(0, len(dataset), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(dataset))))
        batch_items = [(j, dataset[j]) for j in batch_indices]

        # Check which images need captioning
        items_to_process = []
        for idx, item in batch_items:
            if not overwrite:
                if use_memory_storage:
                    # For HuggingFaceImageDataset, check if caption already exists in memory or dataset
                    existing_caption = dataset.get_caption(idx) if hasattr(dataset, "get_caption") else ""
                    if existing_caption:
                        skipped_count += 1
                        continue
                else:
                    # For local folder datasets, check if caption file exists
                    caption_path = Path(item["image_path"]).with_suffix(caption_extension)
                    if caption_path.exists():
                        skipped_count += 1
                        continue
            items_to_process.append((idx, item))

        if not items_to_process:
            continue

        # Load and process images
        images = [_load_image(item["image_path"]) for idx, item in items_to_process]

        with torch.no_grad():
            if prompt:
                inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(device, dtype)
            else:
                inputs = processor(images=images, return_tensors="pt").to(device, dtype)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Save captions
        for (idx, item), caption in zip(items_to_process, captions):
            caption = caption.strip()
            if prefix:
                caption = f"{prefix} {caption}"
            if suffix:
                caption = f"{caption} {suffix}"
            _save_caption(item["image_path"], caption, caption_extension, dataset=dataset, index=idx)
            processed_count += 1

    logger.info("Generated %d captions, skipped %d existing", processed_count, skipped_count)

    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return dataset


@Registry.register_pre_process()
def florence2_caption(
    dataset,
    model_name: Optional[str] = None,
    task: str = "<DETAILED_CAPTION>",
    max_new_tokens: int = 256,
    batch_size: int = 1,
    device: str = "cuda",
    caption_extension: str = ".txt",
    overwrite: bool = False,
    prefix: str = "",
    suffix: str = "",
    use_fp16: bool = True,
    **kwargs,
):
    """Generate captions using Florence-2 model.

    Florence-2 supports multiple tasks:
    - <CAPTION>: Brief caption
    - <DETAILED_CAPTION>: Detailed caption
    - <MORE_DETAILED_CAPTION>: Very detailed caption
    - <OCR>: Optical character recognition
    - <OD>: Object detection

    Args:
        dataset: The dataset to process.
        model_name: Florence-2 model name. Default: "microsoft/Florence-2-large".
        task: Task prompt for Florence-2.
        max_new_tokens: Maximum number of tokens to generate.
        batch_size: Batch size for processing.
        device: Device to run inference on.
        caption_extension: Extension for caption files.
        overwrite: Whether to overwrite existing captions.
        prefix: Prefix to add to all generated captions.
        suffix: Suffix to add to all generated captions.
        use_fp16: Whether to use FP16 for inference.
        **kwargs: Additional generation arguments.

    Returns:
        The dataset with captions generated and saved.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_name = model_name or "microsoft/Florence-2-large"
    logger.info("Loading Florence-2 model: %s", model_name)

    dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()

    # Check if dataset supports in-memory caption storage
    use_memory_storage = hasattr(dataset, "set_caption")

    processed_count = 0
    skipped_count = 0

    for i in range(0, len(dataset), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(dataset))))
        batch_items = [(j, dataset[j]) for j in batch_indices]

        items_to_process = []
        for idx, item in batch_items:
            if not overwrite:
                if use_memory_storage:
                    # For HuggingFaceImageDataset, check if caption already exists
                    existing_caption = dataset.get_caption(idx) if hasattr(dataset, "get_caption") else ""
                    if existing_caption:
                        skipped_count += 1
                        continue
                else:
                    # For local folder datasets, check if caption file exists
                    caption_path = Path(item["image_path"]).with_suffix(caption_extension)
                    if caption_path.exists():
                        skipped_count += 1
                        continue
            items_to_process.append((idx, item))

        if not items_to_process:
            continue

        images = [_load_image(item["image_path"]) for idx, item in items_to_process]

        with torch.no_grad():
            for image, (idx, item) in zip(images, items_to_process):
                inputs = processor(text=task, images=image, return_tensors="pt").to(device, dtype)

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                    **kwargs,
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(generated_text, task=task, image_size=image.size)

                # Extract caption from parsed result
                caption = parsed.get(task, generated_text).strip()
                if prefix:
                    caption = f"{prefix} {caption}"
                if suffix:
                    caption = f"{caption} {suffix}"
                _save_caption(item["image_path"], caption, caption_extension, dataset=dataset, index=idx)
                processed_count += 1

    logger.info("Generated %d captions, skipped %d existing", processed_count, skipped_count)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return dataset
