# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Image datasets for Stable Diffusion LoRA and DreamBooth training."""

import logging
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import Dataset as TorchDataset

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
CAPTION_EXTENSIONS = {".txt", ".caption"}


class ImageFolderDataset(TorchDataset):
    """Generic image folder dataset for SD training.

    Loads images and their associated captions/tags from a directory structure.
    Works with both LoRA and DreamBooth training.

    Expected directory structure:
    data_dir/
    ├── image1.jpg
    ├── image1.txt  (optional caption file)
    ├── image2.png
    ├── image2.txt
    └── ...

    Or with subdirectories (concept folders):
    data_dir/
    ├── concept1/
    │   ├── image1.jpg
    │   ├── image1.txt
    │   └── ...
    └── concept2/
        └── ...
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        recursive: bool = True,
        caption_extension: str = ".txt",
        default_caption: Optional[str] = None,
        shuffle_tags: bool = False,
        tag_separator: str = ", ",
        max_samples: Optional[int] = None,
        # DreamBooth specific
        instance_prompt: Optional[str] = None,
        class_data_dir: Optional[Union[str, Path]] = None,
        class_prompt: Optional[str] = None,
    ):
        """Initialize the image dataset.

        Args:
            data_dir: Directory containing images and caption files.
            recursive: Whether to search subdirectories recursively.
            caption_extension: Extension for caption files (default: ".txt").
            default_caption: Default caption if no caption file exists.
            shuffle_tags: Whether to shuffle tags when loading.
            tag_separator: Separator used between tags in caption files.
            max_samples: Maximum samples to load. None for all.
            instance_prompt: Fixed prompt for all images (DreamBooth style).
            class_data_dir: Directory for class/regularization images.
            class_prompt: Prompt for class images.
        """
        self.data_dir = Path(data_dir).resolve()
        self.recursive = recursive
        self.caption_extension = caption_extension
        self.default_caption = default_caption or ""
        self.shuffle_tags = shuffle_tags
        self.tag_separator = tag_separator
        self.instance_prompt = instance_prompt
        self.class_data_dir = Path(class_data_dir).resolve() if class_data_dir else None
        self.class_prompt = class_prompt

        # Find instance/main images
        self.image_paths = self._find_images(self.data_dir)

        # Find class images if provided
        self.class_image_paths = []
        if self.class_data_dir and self.class_data_dir.exists():
            self.class_image_paths = self._find_images(self.class_data_dir)
            logger.info("Found %d class images", len(self.class_image_paths))

        if max_samples is not None and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]

        logger.info("Found %d images in %s", len(self.image_paths), self.data_dir)

        # Metadata storage (populated by preprocessing steps)
        self.bucket_assignments = {}
        self.buckets = []

    def _find_images(self, directory: Path) -> list[Path]:
        """Find all image files in the directory."""
        image_paths = []

        if self.recursive:
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(directory.rglob(f"*{ext}"))
                image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(directory.glob(f"*{ext}"))
                image_paths.extend(directory.glob(f"*{ext.upper()}"))

        return sorted(image_paths)

    def _get_caption_path(self, image_path: Path) -> Path:
        """Get the caption file path for an image."""
        return image_path.with_suffix(self.caption_extension)

    def _load_caption(self, image_path: Path) -> str:
        """Load caption for an image."""
        # If instance_prompt is set, use it (DreamBooth style)
        if self.instance_prompt:
            return self.instance_prompt

        caption_path = self._get_caption_path(image_path)

        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
            if self.shuffle_tags and self.tag_separator in caption:
                import random

                tags = [t.strip() for t in caption.split(self.tag_separator)]
                random.shuffle(tags)
                caption = self.tag_separator.join(tags)
            return caption

        return self.default_caption

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        """Get an item from the dataset.

        Returns:
            dict with keys:
                - image_path: Path to the image file
                - caption: Caption text for the image
                - concept_folder: Name of the parent folder
                - index: Dataset index
                - class_image_path: (optional) Path to class image
                - class_caption: (optional) Caption for class image
        """
        image_path = self.image_paths[index]
        caption = self._load_caption(image_path)

        # Get concept folder name
        try:
            relative_path = image_path.relative_to(self.data_dir)
            concept_folder = relative_path.parent.name if relative_path.parent.name else ""
        except ValueError:
            concept_folder = ""

        result = {
            "image_path": str(image_path),
            "caption": caption,
            "concept_folder": concept_folder,
            "index": index,
        }

        # Add class image info for DreamBooth
        if self.class_image_paths:
            import random
            class_idx = random.randint(0, len(self.class_image_paths) - 1)
            result["class_image_path"] = str(self.class_image_paths[class_idx])
            result["class_caption"] = self.class_prompt or self.default_caption

        return result

    def update_caption(self, index: int, caption: str, extension: Optional[str] = None) -> None:
        """Update caption for an image.

        Args:
            index: Index of the image.
            caption: New caption text.
            extension: Override caption extension.
        """
        image_path = self.image_paths[index]
        ext = extension or self.caption_extension
        caption_path = image_path.with_suffix(ext)
        caption_path.write_text(caption, encoding="utf-8")

    def get_all_image_paths(self) -> list[str]:
        """Get all image paths as strings."""
        return [str(p) for p in self.image_paths]


# Register the dataset
@Registry.register_dataset()
def image_folder_dataset(
    data_dir: str,
    recursive: bool = True,
    caption_extension: str = ".txt",
    default_caption: Optional[str] = None,
    shuffle_tags: bool = False,
    tag_separator: str = ", ",
    max_samples: Optional[int] = None,
    instance_prompt: Optional[str] = None,
    class_data_dir: Optional[str] = None,
    class_prompt: Optional[str] = None,
) -> ImageFolderDataset:
    """Create an image folder dataset.

    This is the main dataset for SD LoRA and DreamBooth training.

    Args:
        data_dir: Directory containing images.
        recursive: Search subdirectories.
        caption_extension: Extension for caption files.
        default_caption: Default caption if no file exists.
        shuffle_tags: Shuffle tags when loading.
        tag_separator: Separator between tags.
        max_samples: Maximum samples to load.
        instance_prompt: Fixed prompt (DreamBooth).
        class_data_dir: Class images directory (DreamBooth).
        class_prompt: Prompt for class images.

    Returns:
        ImageFolderDataset instance.
    """
    return ImageFolderDataset(
        data_dir=data_dir,
        recursive=recursive,
        caption_extension=caption_extension,
        default_caption=default_caption,
        shuffle_tags=shuffle_tags,
        tag_separator=tag_separator,
        max_samples=max_samples,
        instance_prompt=instance_prompt,
        class_data_dir=class_data_dir,
        class_prompt=class_prompt,
    )


# Keep old name for backwards compatibility
@Registry.register_dataset()
def sd_lora_image_dataset(
    data_dir: str,
    recursive: bool = True,
    caption_extension: str = ".txt",
    default_caption: Optional[str] = None,
    shuffle_tags: bool = False,
    tag_separator: str = ", ",
    max_samples: Optional[int] = None,
) -> ImageFolderDataset:
    """Create an SD LoRA image dataset. Alias for image_folder_dataset."""
    return image_folder_dataset(
        data_dir=data_dir,
        recursive=recursive,
        caption_extension=caption_extension,
        default_caption=default_caption,
        shuffle_tags=shuffle_tags,
        tag_separator=tag_separator,
        max_samples=max_samples,
    )


# Alias for compatibility
SDLoRADataset = ImageFolderDataset
