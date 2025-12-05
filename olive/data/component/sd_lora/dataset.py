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


class HuggingFaceImageDataset(TorchDataset):
    """Wrapper for HuggingFace datasets to work with SD LoRA preprocessing.

    Converts HuggingFace datasets with image/caption columns to the format
    expected by SD LoRA preprocessing (image_path, caption).

    The image column should contain PIL.Image objects, which have a `filename`
    attribute pointing to the cached file path.

    Supports:
    - Datasets with existing captions (via caption_column)
    - Datasets without captions (caption_column=None, use auto_caption to generate)
    - In-memory caption storage for generated captions

    Example:
        >>> from datasets import load_dataset
        >>> hf_ds = load_dataset("linoyts/Tuxemon", split="train")
        >>> ds = HuggingFaceImageDataset(hf_ds, image_column="image", caption_column="prompt")
        >>> item = ds[0]
        >>> print(item["image_path"])  # /home/user/.cache/huggingface/...
        >>> print(item["caption"])     # "a tuxemon cartoon of..."
    """

    def __init__(
        self,
        hf_dataset,
        image_column: str = "image",
        caption_column: Optional[str] = "caption",
    ):
        """Initialize the HuggingFace dataset wrapper.

        Args:
            hf_dataset: HuggingFace dataset with image and caption columns.
            image_column: Column name containing PIL.Image objects.
            caption_column: Column name containing caption text. If None, captions
                must be set via set_caption() or auto_caption preprocessing.
        """
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.caption_column = caption_column

        # Validate image column exists
        if image_column not in hf_dataset.column_names:
            raise ValueError(
                f"Image column '{image_column}' not found in dataset. "
                f"Available columns: {hf_dataset.column_names}"
            )

        # Validate caption column if specified
        if caption_column is not None and caption_column not in hf_dataset.column_names:
            raise ValueError(
                f"Caption column '{caption_column}' not found in dataset. "
                f"Available columns: {hf_dataset.column_names}"
            )

        # Cache for image paths (lazy loaded)
        self._image_paths_cache: dict[int, str] = {}

        # In-memory caption storage (for auto_caption or manual updates)
        # Key: index or image_path, Value: caption string
        self._captions_cache: dict[Union[int, str], str] = {}

        logger.info(
            "Loaded HuggingFace dataset with %d images (image_column=%s, caption_column=%s)",
            len(hf_dataset), image_column, caption_column
        )

        # Metadata storage (populated by preprocessing steps)
        self.bucket_assignments = {}
        self.buckets = []

    def _get_image_path(self, index: int) -> str:
        """Get image path for an index, with caching."""
        if index not in self._image_paths_cache:
            item = self.hf_dataset[index]
            img = item[self.image_column]
            if hasattr(img, "filename") and img.filename:
                self._image_paths_cache[index] = img.filename
            else:
                raise ValueError(
                    f"Image at index {index} does not have a filename attribute. "
                    "This may happen if the dataset is not cached locally."
                )
        return self._image_paths_cache[index]

    @property
    def image_paths(self) -> list[str]:
        """Get all image paths. Builds cache if needed."""
        if len(self._image_paths_cache) < len(self.hf_dataset):
            logger.info("Building image paths cache for %d images...", len(self.hf_dataset))
            for i in range(len(self.hf_dataset)):
                self._get_image_path(i)
        return [self._image_paths_cache[i] for i in range(len(self.hf_dataset))]

    def set_caption(self, index_or_path: Union[int, str], caption: str) -> None:
        """Set caption for an image in memory.

        Args:
            index_or_path: Dataset index or image path.
            caption: Caption text to set.
        """
        self._captions_cache[index_or_path] = caption

    def get_caption(self, index: int) -> str:
        """Get caption for an image.

        Priority:
        1. In-memory cache (by index)
        2. In-memory cache (by image_path)
        3. HuggingFace dataset column
        4. Empty string

        Args:
            index: Dataset index.

        Returns:
            Caption string.
        """
        # Check in-memory cache by index
        if index in self._captions_cache:
            return self._captions_cache[index]

        # Check in-memory cache by image_path
        image_path = self._get_image_path(index)
        if image_path in self._captions_cache:
            return self._captions_cache[image_path]

        # Fall back to HuggingFace dataset column
        if self.caption_column is not None:
            item = self.hf_dataset[index]
            return item.get(self.caption_column, "")

        return ""

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> dict:
        """Get an item from the dataset.

        Returns:
            dict with keys:
                - image_path: Path to the cached image file
                - caption: Caption text for the image
                - index: Dataset index
        """
        image_path = self._get_image_path(index)
        caption = self.get_caption(index)

        return {
            "image_path": image_path,
            "caption": caption,
            "index": index,
        }

    def get_all_image_paths(self) -> list[str]:
        """Get all image paths as strings."""
        return self.image_paths
