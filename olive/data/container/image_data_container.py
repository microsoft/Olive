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
