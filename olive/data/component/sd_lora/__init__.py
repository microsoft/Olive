# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Stable Diffusion LoRA and DreamBooth data components.

This module provides preprocessing components for image training data:

Datasets:
- image_folder_dataset: Generic image folder dataset
- sd_lora_image_dataset: Alias for image_folder_dataset

Preprocessing steps (can be chained):
- image_filtering: Filter images by size, quality, duplicates
- auto_caption: Generate captions using BLIP-2, Florence-2
- auto_tagging: Generate tags using WD14 tagger
- caption_tag_merge: Merge captions and tags
- image_resizing: Resize images with various modes
- aspect_ratio_bucketing: Group images by aspect ratio
- dreambooth_prompt: Generate DreamBooth-style fixed prompts
- generate_class_images: Generate class images for prior preservation

Dataloaders:
- image_bucket_dataloader: Dataloader with bucket batching
- sd_lora_bucket_dataloader: Alias for image_bucket_dataloader

Chain presets:
- sd_lora: Default chain for SD 1.5 LoRA
- sdxl_lora: Default chain for SDXL LoRA
- dreambooth: Default chain for DreamBooth
"""

# Dataset
from olive.data.component.sd_lora.dataset import (
    ImageFolderDataset,
    SDLoRADataset,
    image_folder_dataset,
    sd_lora_image_dataset,
)

# Preprocessing components
from olive.data.component.sd_lora.auto_caption import auto_caption, blip2_caption, florence2_caption
from olive.data.component.sd_lora.auto_tagging import auto_tagging, wd14_tagging
from olive.data.component.sd_lora.caption_tag_merge import caption_tag_merge
from olive.data.component.sd_lora.image_filtering import image_filtering
from olive.data.component.sd_lora.image_resizing import image_resizing
from olive.data.component.sd_lora.aspect_ratio_bucketing import aspect_ratio_bucketing, generate_buckets
from olive.data.component.sd_lora.dreambooth_prompt import dreambooth_prompt, generate_class_images
from olive.data.component.sd_lora.preprocess_chain import (
    sd_lora_preprocess,
    dreambooth_preprocess,
)

# Dataloader
from olive.data.component.sd_lora.dataloader import (
    sd_lora_bucket_dataloader,
    BucketBatchSampler,
    sd_lora_collate_fn,
)

__all__ = [
    # Dataset
    "ImageFolderDataset",
    "SDLoRADataset",
    "image_folder_dataset",
    "sd_lora_image_dataset",
    # Preprocessing
    "auto_caption",
    "blip2_caption",
    "florence2_caption",
    "auto_tagging",
    "wd14_tagging",
    "caption_tag_merge",
    "image_filtering",
    "image_resizing",
    "aspect_ratio_bucketing",
    "generate_buckets",
    "dreambooth_prompt",
    "generate_class_images",
    # Preprocess chains
    "sd_lora_preprocess",
    "dreambooth_preprocess",
    # Dataloader
    "sd_lora_bucket_dataloader",
    "BucketBatchSampler",
    "sd_lora_collate_fn",
]
