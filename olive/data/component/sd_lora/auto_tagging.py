# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


def _load_image(image_path: str):
    """Load an image from path."""
    from PIL import Image

    return Image.open(image_path).convert("RGB")


def _save_tags(image_path: str, tags: list[str]) -> None:
    """Save tags to a text file alongside the image."""
    tag_path = Path(image_path).with_suffix(".txt")
    tag_path.write_text(", ".join(tags), encoding="utf-8")


@Registry.register_pre_process()
def auto_tagging(
    dataset,
    model_name: Optional[str] = None,
    threshold: float = 0.35,
    device: str = "cuda",
    overwrite: bool = False,
    trigger_word: Optional[str] = None,
):
    """Auto-generate tags for images using WD14 tagger.

    Args:
        dataset: The dataset to process.
        model_name: WD14 model name from HuggingFace. Default: "SmilingWolf/wd-swinv2-tagger-v3".
        threshold: Confidence threshold for tags.
        device: Device to run inference on.
        overwrite: Whether to overwrite existing tag files.
        trigger_word: Trigger word to prepend to all tag lists (e.g., "sks").

    Returns:
        The dataset with tags generated and saved.

    """
    return wd14_tagging(
        dataset,
        model_name=model_name,
        threshold=threshold,
        device=device,
        overwrite=overwrite,
        trigger_word=trigger_word,
    )


@Registry.register_pre_process()
def wd14_tagging(
    dataset,
    model_name: Optional[str] = None,
    threshold: float = 0.35,
    general_threshold: Optional[float] = None,
    character_threshold: Optional[float] = None,
    batch_size: int = 1,
    device: str = "cuda",
    overwrite: bool = False,
    trigger_word: Optional[str] = None,
    include_rating: bool = False,
    exclude_tags: Optional[list[str]] = None,
    replace_underscore: bool = True,
    max_tags: Optional[int] = None,
):
    """Generate tags using WaifuDiffusion v14 tagger models.

    Available models (from HuggingFace):
    - SmilingWolf/wd-swinv2-tagger-v3
    - SmilingWolf/wd-convnext-tagger-v3
    - SmilingWolf/wd-vit-tagger-v3
    - SmilingWolf/wd-v1-4-moat-tagger-v2
    - SmilingWolf/wd-v1-4-swinv2-tagger-v2
    - SmilingWolf/wd-v1-4-convnext-tagger-v2
    - SmilingWolf/wd-v1-4-vit-tagger-v2

    Args:
        dataset: The dataset to process.
        model_name: WD14 model name from HuggingFace.
        threshold: Default confidence threshold for tags.
        general_threshold: Threshold for general tags (defaults to threshold).
        character_threshold: Threshold for character tags (defaults to threshold).
        batch_size: Batch size for processing.
        device: Device to run inference on.
        overwrite: Whether to overwrite existing tag files.
        trigger_word: Trigger word to prepend to all tag lists (e.g., "sks").
        include_rating: Whether to include rating tags.
        exclude_tags: Tags to exclude from results.
        replace_underscore: Whether to replace underscores with spaces.
        max_tags: Maximum number of tags to include.

    Returns:
        The dataset with tags generated and saved.

    """
    import numpy as np
    from huggingface_hub import hf_hub_download
    from PIL import Image

    model_name = model_name or "SmilingWolf/wd-swinv2-tagger-v3"
    general_threshold = general_threshold if general_threshold is not None else threshold
    character_threshold = character_threshold if character_threshold is not None else threshold
    exclude_tags = set(exclude_tags or [])

    logger.info("Loading WD14 tagger: %s", model_name)

    # Download model and tags from HuggingFace
    model_path = hf_hub_download(model_name, filename="model.onnx")
    tags_path = hf_hub_download(model_name, filename="selected_tags.csv")

    # Load tags
    import csv

    with open(tags_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tags_data = list(reader)

    # Separate tag categories
    rating_tags = []
    general_tags = []
    character_tags = []

    for i, row in enumerate(tags_data):
        tag_name = row["name"]
        category = int(row["category"])

        if category == 9:  # Rating
            rating_tags.append((i, tag_name))
        elif category == 0:  # General
            general_tags.append((i, tag_name))
        elif category == 4:  # Character
            character_tags.append((i, tag_name))

    # Load ONNX model
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    # Get model input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    target_size = input_shape[1] if len(input_shape) == 4 else 448  # Default WD14 size

    def preprocess_image(image: Image.Image) -> np.ndarray:
        """Preprocess image for WD14 model."""
        # Pad to square
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2

        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))

        # Resize to target size
        padded = padded.resize((target_size, target_size), Image.BICUBIC)

        # Convert to numpy array (BGR format for WD14)
        img_array = np.array(padded, dtype=np.float32)
        return img_array[:, :, ::-1]  # RGB to BGR

    processed_count = 0
    skipped_count = 0

    for i in range(0, len(dataset), batch_size):
        batch_items = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]

        items_to_process = []
        for item in batch_items:
            tag_path = Path(item["image_path"]).with_suffix(".txt")
            if not overwrite and tag_path.exists():
                skipped_count += 1
                continue
            items_to_process.append(item)

        if not items_to_process:
            continue

        # Process batch
        batch_images = []
        for item in items_to_process:
            image = _load_image(item["image_path"])
            processed = preprocess_image(image)
            batch_images.append(processed)

        batch_input = np.stack(batch_images, axis=0)

        # Run inference
        outputs = session.run(None, {input_name: batch_input})[0]

        # Process each result
        for item, probs in zip(items_to_process, outputs):
            tags_result = []

            # Process general tags
            for tag_idx, tag_name in general_tags:
                if probs[tag_idx] >= general_threshold and tag_name not in exclude_tags:
                    if replace_underscore:
                        tag_name = tag_name.replace("_", " ")  # noqa: PLW2901
                    tags_result.append((tag_name, probs[tag_idx]))
            # Process character tags
            for tag_idx, tag_name in character_tags:
                if probs[tag_idx] >= character_threshold and tag_name not in exclude_tags:
                    if replace_underscore:
                        tag_name = tag_name.replace("_", " ")  # noqa: PLW2901
                    tags_result.append((tag_name, probs[tag_idx]))
            # Process rating tags if requested
            if include_rating:
                for tag_idx, tag_name in rating_tags:
                    if probs[tag_idx] >= threshold:
                        tags_result.append((tag_name, probs[tag_idx]))

            # Sort by confidence and apply max_tags limit
            tags_result.sort(key=lambda x: x[1], reverse=True)
            if max_tags is not None:
                tags_result = tags_result[:max_tags]

            # Extract just tag names
            final_tags = [tag for tag, _ in tags_result]

            # Prepend trigger word if specified
            if trigger_word:
                final_tags = [trigger_word, *final_tags]

            # Save tags
            _save_tags(item["image_path"], final_tags)
            processed_count += 1

    logger.info("Generated tags for %d images, skipped %d existing", processed_count, skipped_count)

    return dataset
