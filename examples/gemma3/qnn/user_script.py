# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
import subprocess
import zipfile
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


class GemmaDataset:
    CACHE_DIR = os.getenv("CACHE_DIR", ".cache")

    def __init__(self, model_id: str, first_n: Optional[int] = None):
        self.model_id = model_id
        self.first_n = first_n

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.CACHE_DIR, use_fast=True, trust_remote_code=True
        )

        self.setup_dataset()

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

    def setup_dataset(self):
        # Uses a LlaVA dataset and transforms it to something Gemma-compatible

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

        # Convert the Llava-style conversation to Gemma-style conversation
        self.raw_datasets = self.raw_datasets.map(self._convert_llava_to_gemma_conversation)

        # Extract image details using a lambda to pass the dataset_path
        self.raw_datasets = self.raw_datasets.map(self._extract_image_details)

        # Filter out any images that are not RGB
        self.raw_datasets = self.raw_datasets.filter(lambda x: x["image_mode"] == "RGB")

        # Loads the images and tokenizes the text
        self.raw_datasets = self.raw_datasets.with_transform(self._load_image_and_tokenize)

        for entry in self.raw_datasets:
            logger.error(entry)

    def get_train_dataset(self):
        return self.raw_datasets

    @staticmethod
    def _convert_llava_to_gemma_conversation(entry: dict[str, any]):
        entry["text"] = [
            GemmaDataset._convert_single_llava_to_gemma_conversation(conversation)
            for conversation in entry["conversations"]
        ]
        del entry["conversations"]
        return entry

    @staticmethod
    def _convert_single_llava_to_gemma_conversation(conversation: list[dict[str, str]]) -> dict[str, str | list[dict]]:
        """Convert a single llava-style conversation entry to Gemma-style.

        Examples:
            >>> conversation = {"from": "human", "value": "<image>What are the colors of the bus in the image?"}
            >>> _convert_llava_to_gemma_conversation(conversation)
            {
                'role': 'user',
                'content': [{'type': 'image'}, {'type': 'text', 'text': 'What are the colors of the bus in the image?'}]
            }
            >>> conversation = {"from": "gpt", "value": "The bus in the image is white and red."}
            >>> _convert_llava_to_gemma_conversation(conversation)
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'The bus in the image is white and red.'}]
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

        if "<image>" in text:
            has_image = True
            text = text.replace("<image>", "")
        else:
            has_image = False

        return {
            "role": role,
            "content": (
                [{"type": "image"}, {"type": "text", "text": text}] if has_image else [{"type": "text", "text": text}]
            ),
        }

    def _extract_image_details(self, entry: dict[str, any]):
        """Extract image details from the dataset example.

        Opens the image file and adds image mode information to the example.
        """
        image = PILImage.open(fp=os.path.join(self.image_data_path, entry["image"]))
        entry["image_mode"] = image.mode
        return entry

    def _load_image_and_tokenize(self, entry: dict[str, any]):
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


SHORTCUT_FIRST_N = 256


@Registry.register_dataset()
def gemma_dataset(model_id: str):
    return GemmaDataset(model_id, first_n=SHORTCUT_FIRST_N).get_train_dataset()


@Registry.register_dataset()
def gemma_text_dataset(model_id: str):
    return GemmaDataset(model_id, first_n=SHORTCUT_FIRST_N, filter="text").get_train_dataset


@Registry.register_dataset()
def gemma_vision_dataset(model_id: str):
    return GemmaDataset(model_id, first_n=SHORTCUT_FIRST_N, filter="images").get_train_dataset()
