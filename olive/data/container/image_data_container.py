# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import ClassVar, Optional

from olive.data.constants import DataComponentType, DataContainerType, DatasetType
from olive.data.container.data_container import DataContainer
from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register(DataContainerType.DATA_CONTAINER, name="ImageDataContainer")
class ImageDataContainer(DataContainer):
    """Data container for image model training (LoRA, DreamBooth, fine-tuning).

    Works for SD 1.5, SDXL, Flux, etc. - just set base_resolution:
    - SD 1.5: base_resolution=512 (default)
    - SDXL/Flux: base_resolution=1024

    Supports both local image folders and HuggingFace datasets.

    Usage (HuggingFace dataset with captions):
        config = DataConfig(
            name="my_data",
            type="ImageDataContainer",
            load_dataset_config={
                "type": "huggingface_dataset",
                "params": {
                    "data_name": "linoyts/Tuxemon",
                    "split": "train",
                    "image_column": "image",
                    "caption_column": "prompt"  # optional
                }
            }
        )

    Usage (HuggingFace dataset without captions, use auto_caption):
        config = DataConfig(
            name="my_data",
            type="ImageDataContainer",
            load_dataset_config={
                "type": "huggingface_dataset",
                "params": {
                    "data_name": "my_dataset",
                    "split": "train",
                    "image_column": "image"
                }
            },
            pre_process_data_config={
                "type": "image_lora_preprocess",
                "params": {
                    "steps": {
                        "auto_caption": {},
                        "aspect_ratio_bucketing": {}
                    }
                }
            }
        )

    Usage (Local folder):
        config = DataConfig(
            name="my_data",
            type="ImageDataContainer",
            load_dataset_config={
                "type": "image_folder_dataset",
                "params": {"data_dir": "/path/to/images"}
            },
            pre_process_data_config={
                "type": "image_lora_preprocess",
                "params": {
                    "base_resolution": 1024,
                    "steps": {
                        "auto_caption": {},
                        "aspect_ratio_bucketing": {}
                    }
                }
            }
        )
    """

    default_components_type: ClassVar[dict] = {
        DataComponentType.LOAD_DATASET.value: "image_folder_dataset",
        DataComponentType.PRE_PROCESS_DATA.value: "image_lora_preprocess",
        DataComponentType.POST_PROCESS_DATA.value: "default_post_process_data",
        DataComponentType.DATALOADER.value: "default_dataloader",
    }

    def _is_huggingface_dataset(self) -> bool:
        """Check if the dataset is from HuggingFace."""
        load_type = self.config.load_dataset_config.type
        return load_type == DatasetType.HUGGINGFACE_DATASET

    def _convert_hf_dataset(self, dataset, image_column: str, caption_column: Optional[str]):
        """Convert HuggingFace dataset to SD LoRA format."""
        from olive.data.component.sd_lora.dataset import HuggingFaceImageDataset

        return HuggingFaceImageDataset(dataset, image_column, caption_column)

    def load_dataset(self):
        """Load dataset, extracting ImageDataContainer-specific params first."""
        # Pop image_column and caption_column so they don't get passed to huggingface_dataset
        params = self.config.load_dataset_config.params
        image_column = params.pop("image_column", "image")
        caption_column = params.pop("caption_column", None)

        # Load the raw HuggingFace dataset
        dataset = super().load_dataset()

        # Convert to HuggingFaceImageDataset if needed
        if self._is_huggingface_dataset():
            logger.info(
                "Converting HuggingFace dataset: image_column=%s, caption_column=%s",
                image_column,
                caption_column,
            )
            dataset = self._convert_hf_dataset(dataset, image_column, caption_column)

        return dataset

    def pre_process(self, dataset):
        """Run pre_process with auto resize_images for HuggingFace datasets."""
        # For HuggingFace datasets, images are in cache and can't be pre-resized by user
        # So we need to resize them during preprocessing
        if self._is_huggingface_dataset():
            pre_process_params = self.config.pre_process_params
            # Get or create steps dict
            steps = pre_process_params.get("steps")
            if steps is None:
                # Default steps will be used, create a dict to modify
                steps = {"aspect_ratio_bucketing": {}}
                pre_process_params["steps"] = steps

            # Inject resize_images=True and output_dir for aspect_ratio_bucketing if not explicitly set
            if "aspect_ratio_bucketing" in steps:
                bucket_params = steps["aspect_ratio_bucketing"]
                if "resize_images" not in bucket_params:
                    bucket_params["resize_images"] = True
                    logger.info("HuggingFace dataset detected, enabling resize_images=True for aspect_ratio_bucketing")

                # HuggingFace datasets need an output_dir since images are in cache
                if "output_dir" not in bucket_params and pre_process_params.get("output_dir") is None:
                    from olive.cache import OliveCache

                    # Use Olive's cache directory for resized images
                    cache = OliveCache.from_cache_env()
                    resized_dir = cache.get_cache_dir() / "resized_images" / self.config.name
                    resized_dir.mkdir(parents=True, exist_ok=True)
                    bucket_params["output_dir"] = str(resized_dir)
                    logger.info("HuggingFace dataset detected, using cache output_dir: %s", resized_dir)

        return super().pre_process(dataset)
