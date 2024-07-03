# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import io
import json
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from olive.common.config_utils import ConfigBase
from olive.common.utils import get_credentials, hash_dict
from olive.model.config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class CloudCacheConfig(ConfigBase):
    enable_cloud_cache: bool = True
    account_url: str = "https://olivepublicmodels.blob.core.windows.net"
    contaier_name: str = "olivecachemodels"
    upload_to_cloud: bool = True
    input_model_config: ModelConfig = None


class CloudCacheHelper:
    def __init__(
        self,
        cache_dir: str,
        account_url: str,
        container_name: str,
        input_model_config: ModelConfig = None,
    ):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as exc:
            raise ImportError(
                "Please install azure-storage-blob and azure-identity to use the cloud model cache feature."
            ) from exc
        credential = get_credentials()
        self.container_name = container_name
        self.container_client = ContainerClient(
            account_url=account_url, container_name=self.container_name, credential=credential
        )

        self.cache_dir = cache_dir
        self.output_model_path = Path(cache_dir) / "model"
        self.input_model_config = input_model_config

    def update_model_config(self, cloud_model_path, model_config, input_model_hash):
        logger.info("Updating model config with cloud model path: %s", cloud_model_path)
        model_directory_prefix = f"{input_model_hash}/model"
        blob_list = self.container_client.list_blobs(name_starts_with=model_directory_prefix)

        for blob in blob_list:
            blob_client = self.container_client.get_blob_client(blob)
            local_file_path = self.output_model_path / blob.name[len(model_directory_prefix) + 1 :]
            logger.info("Downloading %s to %s", blob.name, local_file_path)
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

        model_config.config["model_path"] = str(self.output_model_path / cloud_model_path)

        return model_config

    def get_hash_key(self, model_config: ModelConfig, pass_search_point: Dict[str, Any], input_model_hash: str):
        hf_hub_model_commit_id = None
        model_config_copy = deepcopy(model_config)
        if input_model_hash is None:
            from huggingface_hub import repo_info

            if model_config.has_hf_config():
                hf_hub_model_commit_id = repo_info(model_config.get_hf_model_name()).sha
        else:
            model_config_copy.config.pop("model_path", None)
        return hash_dict(
            {
                "hf_hub_model_commit_id": hf_hub_model_commit_id,
                "model_config": model_config_copy.to_json(),
                "pass_search_point": pass_search_point,
                "input_model_hash": input_model_hash,
            }
        )

    def exist_in_cloud_cache(self, output_model_hash: str):
        logger.info("Checking cloud cache for model hash: %s", output_model_hash)
        return self.container_client.get_blob_client(output_model_hash).exists()

    def get_model_config_by_hash_key(self, output_model_hash: str):
        model_config_blob = f"{output_model_hash}/model_config.json"
        model_config_path = Path(self.cache_dir) / output_model_hash
        model_config_path.mkdir(parents=True, exist_ok=True)
        model_config_path = model_config_path / "model_config.json"

        blob_client = self.container_client.get_blob_client(model_config_blob)
        logger.info("Downloading %s to %s", model_config_blob, model_config_path)

        with open(model_config_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

        # model config
        with open(model_config_path) as file:
            model_config_dict = json.load(file)
            return ModelConfig.from_json(model_config_dict)

    def upload_model_to_cloud_cache(self, output_model_hash: str, output_model_config: ModelConfig):
        logger.info("Uploading model to cloud cache.")
        model_path = output_model_config.config.get("model_path")

        if model_path is None:
            logger.error("Model path is not found in the output model config. Upload failed.")
            return

        model_path = Path(model_path)
        model_config_copy = deepcopy(output_model_config)
        model_config_copy.config["model_path"] = None

        # upload model config file
        model_config_bytes = json.dumps(model_config_copy.to_json()).encode()
        with io.BytesIO(model_config_bytes) as data:
            self.container_client.upload_blob(f"{output_model_hash}/model_config.json", data=data, overwrite=False)

        # upload model file
        model_blob = str(Path(output_model_hash) / "model")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path_file = Path(temp_dir) / "model_path.json"
            with open(model_path_file, "w") as file:
                file.write({"model_path": model_path.name})
            with open(model_path_file, "rb") as data:
                self.container_client.upload_blob(f"{model_blob}/model_path.json", data=data, overwrite=False)

        if not model_path.is_dir():
            with open(model_path, "rb") as data:
                self.container_client.upload_blob(f"{model_blob}/{model_path.name}", data=data, overwrite=False)
        else:
            self._upload_dir_to_blob(model_path, f"{model_blob}")

    def _upload_dir_to_blob(self, dir_path, blob_folder_name):
        for item in dir_path.iterdir():
            if item.is_dir():
                self._upload_dir_to_blob(item, f"{blob_folder_name}/{item.name}")
            else:
                blob_name = f"{blob_folder_name}/{item.name}"
                with open(item, "rb") as data:
                    self.container_client.upload_blob(name=blob_name, data=data, overwrite=False)


def check_model_cache(cloud_cache_helper: CloudCacheHelper, input_model_config, pass_search_point, input_model_hash):
    output_model_config = None
    logger.info("Cloud model cache is enabled. Check cloud model cache ...")

    output_model_hash = cloud_cache_helper.get_hash_key(input_model_config, pass_search_point, input_model_hash)
    if cloud_cache_helper.exist_in_cloud_cache(output_model_hash):
        logger.info("Model is found in cloud cache.")
        output_model_config = cloud_cache_helper.get_model_config_by_hash_key(output_model_hash)
    return output_model_config


def update_input_model_config(cloud_cache_helper: CloudCacheHelper, input_model_config, input_model_hash):
    # download model files
    logger.info("Cloud model cache is enabled. Downloading input model files ...")
    if cloud_cache_helper.exist_in_cloud_cache(input_model_hash):
        model_path_blob = f"{input_model_hash}/model/model_path.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path_file = Path(temp_dir) / "model_path.json"
            blob_client = cloud_cache_helper.container_client.get_blob_client(model_path_blob)
            with open(model_path_file, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

            with open(model_path_file) as file:
                model_path = json.load(file)["model_path"]
                cloud_cache_helper.update_model_config(model_path, input_model_config, input_model_hash)


def upload_model_to_cloud(
    cloud_cache_helper: CloudCacheHelper, input_model_config, pass_search_point, input_model_hash, output_model_config
):
    output_model_hash = cloud_cache_helper.get_hash_key(input_model_config, pass_search_point, input_model_hash)
    cloud_cache_helper.upload_model_to_cloud_cache(output_model_hash, output_model_config)
