# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
import subprocess
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


class BaseGemmaDataset(ABC):
    """Abstract base class for Gemma dataset implementations."""

    CACHE_DIR = os.getenv("CACHE_DIR", ".cache")

    def __init__(self, model_id: str, first_n: Optional[int] = None):
        self.model_id = model_id
        self.first_n = first_n
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Initialize attributes that will be set during dataset loading
        self.image_data_path = None
        self.raw_datasets = None

        # Initialize processor components based on subclass requirements
        self._initialize_processor_components()

        self.setup_dataset()

    @abstractmethod
    def _initialize_processor_components(self):
        """Initialize processor components specific to the dataset type."""

    @abstractmethod
    def _process_dataset_entry(self, entry: dict[str, any]):
        """Process a single dataset entry according to the dataset type."""

    def _convert_single_llava_to_gemma_conversation(
        self, conversation: list[dict[str, str]], strip_images: bool = False
    ) -> dict[str, str | list[dict]]:
        """Convert a single llava-style conversation entry to Gemma-style.

        Args:
            conversation: The conversation entry to convert
            strip_images: If True, remove <image> tokens and create text-only content.
                         If False, preserve <image> tokens and create multimodal content.

        Examples:
            >>> conversation = {"from": "human", "value": "<image>What are the colors of the bus in the image?"}
            >>> _convert_single_llava_to_gemma_conversation(conversation, strip_images=False)
            {
                'role': 'user',
                'content': [{'type': 'image'}, {'type': 'text', 'text': 'What are the colors of the bus in the image?'}]
            }
            >>> _convert_single_llava_to_gemma_conversation(conversation, strip_images=True)
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'What are the colors of the bus in the image?'}]
            }

        """
        who = conversation.get("from")
        match who:
            case "human":
                role = "user"
            case "gpt":
                role = "assistant"
            case _:
                raise ValueError(f"Unknown role: {who}")

        text = conversation.get("value")

        if strip_images:
            # Text-only: remove image references completely
            text = text.replace("<image>", "").strip()
            return {
                "role": role,
                "content": [{"type": "text", "text": text}],
            }
        else:
            # Multimodal: preserve image references
            if "<image>" in text:
                has_image = True
                text = text.replace("<image>", "")
            else:
                has_image = False

            return {
                "role": role,
                "content": (
                    [{"type": "image"}, {"type": "text", "text": text}]
                    if has_image
                    else [{"type": "text", "text": text}]
                ),
            }

    def _convert_llava_to_gemma_conversation(self, entry: dict[str, any], strip_images: bool = False):
        """Convert LlaVA-style conversations to Gemma-style."""
        entry["text"] = [
            self._convert_single_llava_to_gemma_conversation(conversation, strip_images=strip_images)
            for conversation in entry["conversations"]
        ]
        del entry["conversations"]
        return entry

    def _download_and_extract_images(self):
        """Download the COCO train2017 image dataset and extract to the cache directory."""
        zip_filename = "train2017.zip"
        zip_path = os.path.join(self.CACHE_DIR, zip_filename)
        extract_path = os.path.join(self.CACHE_DIR, "train2017")

        # Create cache directory if it doesn't exist
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        # Check if images are already downloaded and extracted
        extract_path_obj = Path(extract_path)
        if extract_path_obj.exists() and any(extract_path_obj.iterdir()):
            logger.info("Images already exist at %s", extract_path)
            return extract_path

        # Download the dataset if zip doesn't exist
        if not os.path.exists(zip_path):
            logger.info("Downloading COCO train2017 dataset to %s", zip_path)
            try:
                subprocess.run(
                    [
                        "wget",
                        "https://images.cocodataset.org/zips/train2017.zip",
                        "--no-check-certificate",
                        "-O",
                        zip_path,
                    ],
                    check=True,
                    cwd=self.CACHE_DIR,
                )
                logger.info("Download completed successfully")
            except subprocess.CalledProcessError:
                logger.exception("Failed to download dataset")
                raise
            except FileNotFoundError:
                logger.exception("wget command not found. Please install wget or use an alternative download method.")
                raise

        # Extract the zip file
        logger.info("Extracting %s to %s", zip_path, self.CACHE_DIR)
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.CACHE_DIR)
            logger.info("Extraction completed successfully")
        except zipfile.BadZipFile:
            logger.exception("Failed to extract zip file")
            # Remove corrupted zip file so it can be re-downloaded
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise

        return extract_path

    def _load_base_dataset(self):
        """Load the base LlaVA dataset."""
        # Issue with Arrow leads to errors when using load_dataset directly on liuhaotian/LLaVA-Instruct-150K
        file_path = hf_hub_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            filename="llava_instruct_80k.json",
            repo_type="dataset",
            cache_dir=self.CACHE_DIR,
        )

        self.image_data_path = self._download_and_extract_images()
        self.raw_datasets = load_dataset("json", data_files=[file_path], split="train")

        # Limit data processing to the first_n rows
        self.raw_datasets = self.raw_datasets if self.first_n is None else self.raw_datasets.select(range(self.first_n))

    def _extract_image_details(self, entry: dict[str, any]):
        """Extract image details from the dataset example.

        Opens the image file and adds image mode information to the example.
        """
        image = PILImage.open(fp=os.path.join(self.image_data_path, entry["image"]))
        entry["image_mode"] = image.mode
        return entry

    def setup_dataset(self):
        """Set up the dataset with common preprocessing steps."""
        self._load_base_dataset()

        # Extract image details
        self.raw_datasets = self.raw_datasets.map(self._extract_image_details)

        # Filter out any images that are not RGB
        self.raw_datasets = self.raw_datasets.filter(lambda x: x["image_mode"] == "RGB")

        # Apply dataset-specific processing
        logger.error(self.raw_datasets[0])
        logger.error(self.raw_datasets[1])

        self.raw_datasets = self.raw_datasets.with_transform(self._process_dataset_entry)

        logger.error(self.raw_datasets[0])
        logger.error(self.raw_datasets[1])

    def get_dataset(self):
        """Return the processed dataset."""
        return self.raw_datasets


class GemmaMultimodalDataset(BaseGemmaDataset):
    """Dataset for full E2E Gemma 3 multi-modal model including both image and text."""

    def _initialize_processor_components(self):
        """Initialize tokenizer for multimodal processing."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.CACHE_DIR, use_fast=True, trust_remote_code=True
        )

    def setup_dataset(self):
        """Set up the multimodal dataset with text conversation conversion."""
        self._load_base_dataset()

        # Convert the Llava-style conversation to Gemma-style conversation (preserve images)
        self.raw_datasets = self.raw_datasets.map(
            lambda entry: self._convert_llava_to_gemma_conversation(entry, strip_images=False)
        )

        # Extract image details
        self.raw_datasets = self.raw_datasets.map(self._extract_image_details)

        # Filter out any images that are not RGB
        self.raw_datasets = self.raw_datasets.filter(lambda x: x["image_mode"] == "RGB")

        # Apply multimodal processing
        self.raw_datasets = self.raw_datasets.with_transform(self._process_dataset_entry)

    def _process_dataset_entry(self, entry: dict[str, any]):
        """Load image and tokenize the conversation for model input.

        Args:
            entry: Dataset entry containing text conversation and image path

        Returns:
            Tokenized inputs ready for model processing

        """
        inputs = self.processor.apply_chat_template(
            entry["text"][0], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"][0]
        return inputs


class GemmaTextOnlyDataset(BaseGemmaDataset):
    """Dataset for only the text portion of the Gemma 3 model."""

    def _initialize_processor_components(self):
        """Initialize tokenizer for text-only processing."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.CACHE_DIR, use_fast=True, trust_remote_code=True
        )

    def setup_dataset(self):
        """Set up the text-only dataset with conversation conversion."""
        self._load_base_dataset()

        # Convert the Llava-style conversation to Gemma-style conversation (strip images)
        self.raw_datasets = self.raw_datasets.map(
            lambda entry: self._convert_llava_to_gemma_conversation(entry, strip_images=True)
        )

        # Extract image details (still needed for filtering)
        self.raw_datasets = self.raw_datasets.map(self._extract_image_details)

        # Filter out any images that are not RGB
        self.raw_datasets = self.raw_datasets.filter(lambda x: x["image_mode"] == "RGB")

        # Apply text-only processing
        self.raw_datasets = self.raw_datasets.with_transform(self._process_dataset_entry)

    def _process_dataset_entry(self, entry: dict[str, any]):
        """Extract and tokenize only the text content.

        Args:
            entry: Dataset entry containing text conversation

        Returns:
            Tokenized text inputs ready for model processing

        """
        # Apply chat template without images, text-only
        inputs = self.tokenizer.apply_chat_template(
            entry["text"][0], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension


class GemmaVisionOnlyDataset(BaseGemmaDataset):
    """Dataset for only the vision tower of the Gemma 3 model."""

    def _initialize_processor_components(self):
        """No additional components needed for vision-only processing."""

    def _process_dataset_entry(self, entry: dict[str, any]):
        """Load image and extract only pixel_values for vision-only processing."""
        # Load and process the image
        logger.error("PROCESSING IMAGE")
        image = PILImage.open(fp=os.path.join(self.image_data_path, entry["image"][0]))

        # Process image to get pixel_values
        inputs = self.processor(text="<start_of_image>", images=image, return_tensors="pt")

        # Return only pixel_values
        return {"pixel_values": inputs["pixel_values"]}


# Remove this when submitting for review
SHORTCUT_FIRST_N = 256


@Registry.register_dataset()
def gemma_dataset(model_id: str):
    """Full E2E Gemma 3 multi-modal dataset (image + text)."""
    return GemmaMultimodalDataset(model_id, first_n=SHORTCUT_FIRST_N).get_dataset()


@Registry.register_dataset()
def gemma_text_dataset(model_id: str):
    """Text-only Gemma 3 dataset."""
    return GemmaTextOnlyDataset(model_id, first_n=SHORTCUT_FIRST_N).get_dataset()


@Registry.register_dataset()
def gemma_vision_dataset(model_id: str):
    """Vision-only Gemma 3 dataset."""
    return GemmaVisionOnlyDataset(model_id, first_n=SHORTCUT_FIRST_N).get_dataset()
