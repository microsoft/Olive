# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
import io
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from olive.common.config_utils import ConfigBase
from olive.common.utils import get_credentials, hash_dict
from olive.engine.config import FAILED_CONFIG, PRUNED_CONFIGS
from olive.model.config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class CloudCacheConfig(ConfigBase):
    enable_cloud_cache: bool = True
    account_url: str = "https://olivepublicmodels.blob.core.windows.net"
    contaier_name: str = "olivecachemodels"
    upload_to_cloud: bool = True
    input_model_config: ModelConfig = None
    hf_cache_path: str = os.environ.get("HF_HOME", None) or str(Path("~/.cache/huggingface").expanduser())


class CloudModelInvalidStatus(str, Enum):
    NO_MODEL_FILE = "NO_MODEL_FILE"
    FAILED_CONFIG = "FAILED_CONFIG"

class CloudCacheHelper:
    def __init__(
        self,
        cache_dir: str,
        account_url: str,
        container_name: str,
        hf_cache_path: str,
        input_model_config: ModelConfig = None,
    ):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError:
            raise ImportError(
                "Please install azure-storage-blob and azure-identity to use the cloud model cache feature."
            )
        credential = get_credentials()
        self.container_name = container_name
        self.container_client = ContainerClient(
            account_url=account_url, container_name=self.container_name, credential=credential
        )

        self.map_blob = "model_map.json"
        self.local_map_path = Path(cache_dir) / "model_map.json"
        self.local_map_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.output_model_path = Path(cache_dir) / "model"
        self.cloud_cache_map = self.download_model_cache_map()
        self.input_model_config = input_model_config
        self.hf_cache_path = hf_cache_path

    def update_model_config(self, cloud_model_path, model_config, input_model_hash):
        logger.info(f"Updating model config with cloud model path: {cloud_model_path}")
        model_directory_prefix = f"{input_model_hash}/model"
        blob_list = self.container_client.list_blobs(name_starts_with=model_directory_prefix)

        for blob in blob_list:
            blob_client = self.container_client.get_blob_client(blob)
            local_file_path = self.output_model_path / blob.name[len(model_directory_prefix) + 1 :]
            logger.info(f"Downloading {blob.name} to {local_file_path}")
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

        model_config.config["model_path"] = str(self.output_model_path / cloud_model_path)

        return model_config

    def download_model_cache_map(self):
        logger.info(f"Downloading model cache map from {self.map_blob}")
        blob_client = self.container_client.get_blob_client(self.map_blob)

        if not blob_client.exists():
            return {}

        self.local_map_path.unlink(missing_ok=True)
        self.local_map_path.touch()

        with open(self.local_map_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

        with open(self.local_map_path) as file:
            map_dict = json.load(file)
            
        return map_dict

    def get_hash_key(self, model_config: ModelConfig, pass_search_point: Dict[str, Any], input_model_hash: str):
        hf_hub_model_commit_id = None
        if input_model_hash is None:
            model_config_copy = self.input_model_config
            model_path = model_config_copy.config.get("model_path")
            if model_path is None and model_config_copy.config.get("hf_config") is not None:
                # hf model, get model commit from hf cache
                cache_dir = Path(self.hf_cache_path)
                model_cache_dir = Path(cache_dir) / "hub"
                model_name = f"models--{model_config_copy.config['hf_config']['model_name'].replace('/', '--')}"
                commit_path = model_cache_dir / model_name / "refs" / "main"
                if not commit_path.exists():
                    raise Exception(
                        "Huggingface cache not found. Please specify 'hf_cache_path' in 'enable_cloud_cache' config."
                    )
                with open(commit_path) as file:
                    hf_hub_model_commit_id = file.read().strip()
        else:
            model_config_copy = deepcopy(model_config)
            if model_config_copy.config.get("model_path") is not None:
                model_config_copy.config["model_path"] = None
        return hash_dict(
            {
                "hf_hub_model_commit_id": hf_hub_model_commit_id,
                "model_config": model_config_copy.to_json(),
                "pass_search_point": pass_search_point,
                "input_model_hash": input_model_hash,
            }
        )

    def exist_in_model_cache_map(self, output_model_hash: str):
        model_path = self.cloud_cache_map.get(output_model_hash)
        if not model_path:
            logger.warning("Model cache not found in the cloud.")
            return None
        return model_path

    def get_model_config_by_hash_key(self, output_model_hash: str):
        model_config_blob = f"{output_model_hash}/model_config.json"
        model_config_path = Path(self.cache_dir) / output_model_hash
        model_config_path.mkdir(parents=True, exist_ok=True)
        model_config_path = model_config_path / "model_config.json"

        blob_client = self.container_client.get_blob_client(model_config_blob)
        logger.info(f"Downloading {model_config_blob} to {model_config_path}")

        with open(model_config_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

        # model config
        with open(model_config_path) as file:
            model_config_dict = json.load(file)
            return ModelConfig.from_json(model_config_dict)

    def update_model_cache_map(self, output_model_hash, cloud_model_path):
        logger.info("Updating model cache map.")
        if not self.local_map_path.exists():
            self.local_map_path.touch()
            with open(self.local_map_path, "w") as file:
                file.write(b"{}")
        with open(self.local_map_path, "r+b") as file:
            self.cloud_cache_map = json.load(file)
        self.cloud_cache_map[output_model_hash] = cloud_model_path

        cache_map_bytes = json.dumps(self.cloud_cache_map).encode()
        with io.BytesIO(cache_map_bytes) as data:
            self.container_client.upload_blob(self.map_blob, data=data, overwrite=True)

    def upload_model_to_cloud_cache(self, output_model_hash: str, output_model_config: ModelConfig):
        logger.info("Uploading model to cloud cache.")
        model_path = output_model_config.config.get("model_path")

        if model_path is None:
            logger.error("Model path is not found in the output model config.")
            return str(CloudModelInvalidStatus.NO_MODEL_FILE)

        model_path = Path(model_path)
        model_config_copy = deepcopy(output_model_config)
        model_config_copy.config["model_path"] = None

        # upload model config file
        model_config_bytes = json.dumps(model_config_copy.to_json()).encode()
        with io.BytesIO(model_config_bytes) as data:
            self.container_client.upload_blob(f"{output_model_hash}/model_config.json", data=data, overwrite=True)

        # upload model file
        model_blob = str(Path(output_model_hash) / "model")
        if not model_path.is_dir():
            with open(model_path, "rb") as data:
                self.container_client.upload_blob(f"{model_blob}/{model_path.name}", data=data, overwrite=True)
        else:
            self._upload_dir_to_blob(model_path, f"{model_blob}")
        return model_path.name

    def _upload_dir_to_blob(self, dir_path, blob_folder_name):
        for item in dir_path.iterdir():
            if item.is_dir():
                self._upload_dir_to_blob(item, f"{blob_folder_name}/{item.name}")
            else:
                blob_name = f"{blob_folder_name}/{item.name}"
                with open(item, "rb") as data:
                    self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)


def check_model_cache(cloud_cache_helper: CloudCacheHelper, input_model_config, pass_search_point, input_model_hash):
    pass_run_locally = True
    output_model_config = None
    logger.info("Cloud model cache is enabled. Check cloud model cache ...")

    output_model_hash = cloud_cache_helper.get_hash_key(
        input_model_config, pass_search_point, input_model_hash
    )
    cloud_model_path = cloud_cache_helper.exist_in_model_cache_map(cloud_cache_helper.cloud_cache_map, output_model_hash)
    if cloud_model_path is not None:
        if cloud_model_path == CloudModelInvalidStatus.FAILED_CONFIG:
            logger.info("Model is pruned in cloud cache.")
            output_model_config = FAILED_CONFIG
            pass_run_locally = False
        elif cloud_model_path == CloudModelInvalidStatus.NO_MODEL_FILE:
            logger.info("Model file is not found in cloud cache.")
        else:
            logger.info("Model is found in cloud cache.")
            output_model_config = cloud_cache_helper.get_model_config_by_hash_key(output_model_hash)
            pass_run_locally = False
    return pass_run_locally, output_model_config

def update_input_model_config(cloud_cache_helper: CloudCacheHelper, input_model_config, input_model_hash):
    # download model files
    logger.info("Cloud model cache is enabled. Downloading input model files ...")
    cloud_model_path = cloud_cache_helper.exist_in_model_cache_map(cloud_cache_helper.cloud_cache_map, input_model_hash)
    if cloud_model_path is not None:
        cloud_cache_helper.update_model_config(cloud_model_path, input_model_config, input_model_hash)
        
def upload_model_to_cloud(cloud_cache_helper: CloudCacheHelper, input_model_config, pass_search_point, input_model_hash, output_model_config):
    output_model_hash = cloud_cache_helper.get_hash_key(
        input_model_config, pass_search_point, input_model_hash
    )
    cloud_model_path = "FAILED_CONFIG"
    if output_model_config not in PRUNED_CONFIGS:
        cloud_model_path = cloud_cache_helper.upload_model_to_cloud_cache(
            output_model_hash, output_model_config
        )
    cloud_cache_helper.update_model_cache_map(output_model_hash, cloud_model_path)