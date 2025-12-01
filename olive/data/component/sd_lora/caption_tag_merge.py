# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Caption and tag merging component for Stable Diffusion LoRA training."""

import logging
import random
from pathlib import Path
from typing import Optional

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_pre_process()
def caption_tag_merge(
    dataset,
    caption_extension: str = ".caption",
    tag_extension: str = ".tags",
    output_extension: str = ".txt",
    merge_mode: str = "caption_first",
    separator: str = ", ",
    tag_separator: str = ", ",
    shuffle_tags: bool = False,
    keep_n_tags: Optional[int] = None,
    drop_n_tags: int = 0,
    drop_probability: float = 0.0,
    caption_prefix: str = "",
    caption_suffix: str = "",
    tag_prefix: str = "",
    tag_suffix: str = "",
    trigger_word: Optional[str] = None,
    trigger_word_position: str = "start",
    overwrite: bool = False,
    deduplicate_tags: bool = True,
    normalize_tags: bool = True,
    **kwargs,
):
    """Merge captions and tags into final training prompts.

    This component combines auto-generated captions with tags to create
    the final text prompts used for LoRA training.

    Merge modes:
    - caption_first: "caption, tag1, tag2, ..."
    - tags_first: "tag1, tag2, ..., caption"
    - caption_only: Use only the caption
    - tags_only: Use only the tags
    - interleave: Randomly interleave caption words with tags
    - template: Use a custom template with {caption} and {tags} placeholders

    Args:
        dataset: The dataset to process.
        caption_extension: Extension for caption files.
        tag_extension: Extension for tag files.
        output_extension: Extension for output merged files.
        merge_mode: How to merge captions and tags.
        separator: Separator between caption and tags.
        tag_separator: Separator between individual tags.
        shuffle_tags: Whether to shuffle tag order.
        keep_n_tags: Keep only the first N tags (after shuffling if enabled).
        drop_n_tags: Randomly drop N tags per sample.
        drop_probability: Probability of dropping each individual tag.
        caption_prefix: Prefix to add to caption.
        caption_suffix: Suffix to add to caption.
        tag_prefix: Prefix to add to entire tag string.
        tag_suffix: Suffix to add to entire tag string.
        trigger_word: Trigger word/phrase for the LoRA concept.
        trigger_word_position: Where to place trigger word ("start", "end", "caption_start").
        overwrite: Whether to overwrite existing output files.
        deduplicate_tags: Remove duplicate tags.
        normalize_tags: Normalize tag formatting (lowercase, strip whitespace).
        **kwargs: Additional arguments.

    Returns:
        The dataset with merged captions saved.
    """
    processed_count = 0
    skipped_count = 0

    for i in range(len(dataset)):
        item = dataset[i]
        image_path = Path(item["image_path"])

        output_path = image_path.with_suffix(output_extension)
        if not overwrite and output_path.exists():
            skipped_count += 1
            continue

        # Load caption
        caption_path = image_path.with_suffix(caption_extension)
        caption = ""
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()

        # Load tags
        tag_path = image_path.with_suffix(tag_extension)
        tags = []
        if tag_path.exists():
            tag_content = tag_path.read_text(encoding="utf-8").strip()
            if tag_content:
                tags = [t.strip() for t in tag_content.split(tag_separator) if t.strip()]

        # Normalize tags if requested
        if normalize_tags:
            tags = [t.lower().strip() for t in tags]
            caption = caption.strip()

        # Deduplicate tags
        if deduplicate_tags:
            seen = set()
            unique_tags = []
            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower not in seen:
                    seen.add(tag_lower)
                    unique_tags.append(tag)
            tags = unique_tags

        # Shuffle tags if requested
        if shuffle_tags and tags:
            random.shuffle(tags)

        # Apply tag dropping
        if drop_probability > 0 and tags:
            tags = [t for t in tags if random.random() > drop_probability]

        if drop_n_tags > 0 and len(tags) > drop_n_tags:
            drop_indices = random.sample(range(len(tags)), drop_n_tags)
            tags = [t for i, t in enumerate(tags) if i not in drop_indices]

        # Apply keep_n_tags limit
        if keep_n_tags is not None and len(tags) > keep_n_tags:
            tags = tags[:keep_n_tags]

        # Apply prefixes and suffixes
        if caption:
            if caption_prefix:
                caption = f"{caption_prefix} {caption}"
            if caption_suffix:
                caption = f"{caption} {caption_suffix}"

        if tags:
            tag_string = tag_separator.join(tags)
            if tag_prefix:
                tag_string = f"{tag_prefix} {tag_string}"
            if tag_suffix:
                tag_string = f"{tag_string} {tag_suffix}"
        else:
            tag_string = ""

        # Merge based on mode
        if merge_mode == "caption_first":
            if caption and tag_string:
                merged = f"{caption}{separator}{tag_string}"
            elif caption:
                merged = caption
            else:
                merged = tag_string
        elif merge_mode == "tags_first":
            if tag_string and caption:
                merged = f"{tag_string}{separator}{caption}"
            elif tag_string:
                merged = tag_string
            else:
                merged = caption
        elif merge_mode == "caption_only":
            merged = caption
        elif merge_mode == "tags_only":
            merged = tag_string
        elif merge_mode == "interleave":
            # Interleave caption words with tags
            caption_words = caption.split() if caption else []
            merged_parts = []
            tag_idx = 0
            for word in caption_words:
                merged_parts.append(word)
                if tag_idx < len(tags) and random.random() > 0.5:
                    merged_parts.append(tags[tag_idx])
                    tag_idx += 1
            # Add remaining tags
            merged_parts.extend(tags[tag_idx:])
            merged = " ".join(merged_parts)
        elif merge_mode == "template":
            # Use template from kwargs
            template = kwargs.get("template", "{caption}, {tags}")
            merged = template.format(caption=caption, tags=tag_string)
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")

        # Add trigger word
        if trigger_word:
            if trigger_word_position == "start":
                merged = f"{trigger_word}, {merged}" if merged else trigger_word
            elif trigger_word_position == "end":
                merged = f"{merged}, {trigger_word}" if merged else trigger_word
            elif trigger_word_position == "caption_start":
                # Insert trigger word at the beginning of the caption portion
                if merge_mode == "caption_first":
                    merged = f"{trigger_word}, {merged}" if merged else trigger_word
                elif merge_mode == "tags_first" and caption:
                    merged = merged.replace(caption, f"{trigger_word}, {caption}")
                else:
                    merged = f"{trigger_word}, {merged}" if merged else trigger_word

        # Save merged result
        output_path.write_text(merged.strip(), encoding="utf-8")
        processed_count += 1

    logger.info("Merged %d caption/tag files, skipped %d existing", processed_count, skipped_count)

    return dataset
