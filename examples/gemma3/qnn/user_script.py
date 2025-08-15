# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import numpy as np
import os
import subprocess
import zipfile
import torch

from huggingface_hub import hf_hub_download
from typing import Optional

from transformers import pipeline
import requests
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import AutoConfig, AutoTokenizer
from itertools import chain
from torch.utils.data import DataLoader, Dataset
from datasets import IterableDataset, load_dataset
from transformers import default_data_collator
from PIL import Image as PILImage
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import make_nested_list_of_images

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import CLIPProcessor

from olive.data.registry import Registry
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)



def get_gemma3_dataset(tokenzier, processor, data_files, dataset_path, cache_dir):
    def _map1(example):
        example['text'] = [_convert_one_conversation(conversation=conversation) for conversation in
                            example['conversations']]
        return example

    def _map2(example):
        image = PILImage.open(fp=os.path.join(dataset_path, example["image"]))
        example['image_mode'] = image.mode
        return example

    def _load_image_and_tokenize(example):
        # try:
            #print(example['text'])
            inputs = processor.apply_chat_template(example['text'][0],
                                                   add_generation_prompt=True, tokenize=True,
                                                   return_tensors="pt", return_dict=True)
            # print("image=", example["image"][0])
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
            # inputs.update({"pixel_values": torch.tensor(processor(text="<start_of_image>", images=PILImage.open(fp=os.path.join(dataset_path, example["image"][0]))).pixel_values).unsqueeze(0)})
            #print(inputs.keys())
            inputs["input_ids"] = inputs["input_ids"][0]
            #print(inputs["input_ids"])
            return inputs
        
        # except Exception as e:
        #     print(f"Skipping example due to error: {e}")
        #     return None

    
    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir, split='train')

    dataset = dataset.map(_map1)
    dataset = dataset.map(_map2)
    
    dataset = dataset.filter(lambda x: x["image_mode"] == 'RGB')

    return dataset.with_transform(_load_image_and_tokenize)

class GemmaDataset:

    CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
     
    def __init__(self, model_id: str, first_n: Optional[int] = None):
        self.model_id = model_id
        self.first_n = first_n
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.CACHE_DIR, use_fast=True, trust_remote_code=True)

        self.setup_dataset()

    def _download_and_extract_images(self):
        """
        Downloads the coco train2017 image dataset and extracts them to the cache directory
        """
        zip_filename = "train2017.zip"
        zip_path = os.path.join(self.CACHE_DIR, zip_filename)
        extract_path = os.path.join(self.CACHE_DIR, "train2017")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        # Check if images are already downloaded and extracted
        if os.path.exists(extract_path) and os.listdir(extract_path):
            logger.info(f"Images already exist at {extract_path}")
            return extract_path
        
        # Download the dataset if zip doesn't exist
        if not os.path.exists(zip_path):
            logger.info(f"Downloading COCO train2017 dataset to {zip_path}")
            try:
                subprocess.run([
                    "wget", 
                    "https://images.cocodataset.org/zips/train2017.zip",
                    "--no-check-certificate",
                    "-O", zip_path
                ], check=True, cwd=self.CACHE_DIR)
                logger.info("Download completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download dataset: {e}")
                raise
            except FileNotFoundError:
                logger.error("wget command not found. Please install wget or use an alternative download method.")
                raise
        
        # Extract the zip file
        logger.info(f"Extracting {zip_path} to {self.CACHE_DIR}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.CACHE_DIR)
            logger.info("Extraction completed successfully")
        except zipfile.BadZipFile as e:
            logger.error(f"Failed to extract zip file: {e}")
            # Remove corrupted zip file so it can be re-downloaded
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise
        
        return extract_path

    def setup_dataset(self):
         # Uses a LlaVA dataset and transforms it to something Gemma-compatible

         # Issue with Arrow leads to errors when using load_dataset directly on liuhaotian/LLaVA-Instruct-150K
         file_path = hf_hub_download(repo_id="liuhaotian/LLaVA-Instruct-150K", filename="llava_instruct_80k.json", repo_type="dataset", cache_dir=self.CACHE_DIR)

         self.image_data_path = self._download_and_extract_images()
         self.raw_datasets = load_dataset("json", data_files=[file_path], split="train")

         # Limit data processing to the first_n rows
         self.raw_datasets = self.raw_datasets if self.first_n is None else self.raw_datasets.select(range(self.first_n))

         # Convert the Llava-style conversation to Gemma-style conversation
         self.raw_datasets = self.raw_datasets.map(self._convert_llava_to_gemma_conversation)

         # Extract image details using a lambda to pass the dataset_path
         self.raw_datasets = self.raw_datasets.map(self._extract_image_details)

         # Filter out any images that are not RGB
         self.raw_datasets = self.raw_datasets.filter(lambda x: x["image_mode"] == 'RGB')

         # Loads the images and tokenizes the text
         self.raw_datasets = self.raw_datasets.with_transform(self._load_image_and_tokenize)

         for entry in self.raw_datasets:
             logger.error(entry)

    def get_train_dataset(self):
        return self.raw_datasets
    
    @staticmethod
    def _convert_llava_to_gemma_conversation(entry: dict[str, any]):
        entry['text'] = [GemmaDataset._convert_single_llava_to_gemma_conversation(conversation) for conversation in entry["conversations"]]
        del entry['conversations']
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
        """
        Extract image details from the dataset example.
        Opens the image file and adds image mode information to the example.
        """
        image = PILImage.open(fp=os.path.join(self.image_data_path, entry["image"]))
        entry['image_mode'] = image.mode
        return entry

    def _load_image_and_tokenize(self, entry: dict[str, any]):
        """
        Load image and tokenize the conversation for model input.
        
        Args:
            entry: Dataset entry containing text conversation and image path
            
        Returns:
            Tokenized inputs ready for model processing
        """
        inputs = self.processor.apply_chat_template(entry['text'][0],
                                                   add_generation_prompt=True, tokenize=True,
                                                   return_tensors="pt", return_dict=True)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"][0]
        return inputs


@Registry.register_dataset()
def gemma_dataset(model_id: str):
    return GemmaDataset(model_id, first_n=200).get_train_dataset()
