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
from typing import Any, Dict, Optional

from olive.common.config_utils import ConfigBase
from olive.common.utils import get_credentials, hash_dict
from olive.model.config.model_config import ModelConfig
from olive.resource_path import create_resource_path

logger = logging.getLogger(__name__)


class CloudCacheConfig(ConfigBase):
    enable_cloud_cache: bool = True
    account_url: str = "https://olivepublicmodels.blob.core.windows.net"
    container_name: str = "olivecachemodels"
    upload_to_cloud: bool = True
    input_model_config: ModelConfig = None


class CloudCacheHelper:
    def __init__(
        self,
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

        self.model_path_map = "model_path.json"
        self.input_model_config = input_model_config

    def update_model_config(
        self,
        cloud_model_path: str,
        cloud_adapter_path: str,
        model_config: ModelConfig,
        input_model_hash: str,
        output_model_path: Path,
    ) -> ModelConfig:
        """Download model files from cloud cache and update model config.

        Args:
            cloud_model_path (str): Model path stored in cloud cache
            cloud_adapter_path (str): Adapter path stored in cloud cache
            model_config (ModelConfig): Model config to be updated
            input_model_hash (str): Input model hash
            output_model_path (Path): Output model path

        Returns:
            ModelConfig: updated model config

        """
        logger.info("Updating model config with cloud model path: %s", cloud_model_path)
        model_directory_prefix = f"{input_model_hash}/model"
        blob_list = self.container_client.list_blobs(name_starts_with=model_directory_prefix)
        self._download_blob_list(blob_list, model_directory_prefix, output_model_path)

        adapter_directory_prefix = f"{input_model_hash}/adapter"
        blob_list = self.container_client.list_blobs(name_starts_with=adapter_directory_prefix)
        self._download_blob_list(blob_list, adapter_directory_prefix, output_model_path, "adapter")

        if model_config.type.lower() == "hfmodel" and is_hf_repo_exist(cloud_model_path):
            model_config.config["model_path"] = cloud_model_path
        else:
            model_config.config["model_path"] = str(output_model_path / cloud_model_path)

        if model_config.type.lower() == "hfmodel" and cloud_adapter_path:
            model_config.config["adapter_path"] = str(output_model_path / cloud_adapter_path)

        return model_config

    def get_hash_key(
        self, model_config: ModelConfig, pass_search_point: Dict[str, Any], input_model_hash: Optional[str]
    ):
        """Get hash key from input model config, pass search point, and input model hash."""
        try:
            from huggingface_hub import repo_info
        except ImportError:
            logger.exception(
                "huggingface_hub is not installed. "
                "Please install huggingface_hub to use the cloud model cache feature for Huggingface model."
            )

        hf_hub_model_commit_id = None
        model_config_copy = deepcopy(model_config)
        if (
            input_model_hash is None
            and model_config.type.lower() == "hfmodel"
            and create_resource_path(model_config.config["model_path"]).is_string_name()
        ):
            hf_hub_model_commit_id = repo_info(model_config.config["model_path"]).sha
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

    def exist_in_cloud_cache(self, output_model_hash: str) -> bool:
        logger.info("Checking cloud cache for model hash: %s", output_model_hash)
        return any(self.container_client.list_blobs(output_model_hash))

    def get_model_config_by_hash_key(self, output_model_hash: str, output_model_path: Path) -> ModelConfig:
        """Get model config from cloud cache by hash key."""
        model_config_blob = f"{output_model_hash}/model_config.json"
        model_config_path = output_model_path / "model_config.json"

        blob_client = self.container_client.get_blob_client(model_config_blob)
        logger.info("Downloading %s to %s", model_config_blob, model_config_path)

        with open(model_config_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

        # model config
        with open(model_config_path) as file:
            model_config_dict = json.load(file)
            return ModelConfig.from_json(model_config_dict)

    def get_path_from_cloud(self, output_model_hash: str) -> tuple:
        """Get stored model path and adapter path from cloud cache."""
        blob = f"{output_model_hash}/{self.model_path_map}"
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path_file = Path(temp_dir) / self.model_path_map
            blob_client = self.container_client.get_blob_client(blob)
            with open(model_path_file, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

            with open(model_path_file) as file:
                model_path_map = json.load(file)
                model_path = model_path_map["model_path"]
                adapter_path = model_path_map.get("adapter_path")
                return model_path, adapter_path

    def upload_model_to_cloud_cache(self, output_model_hash: str, output_model_config: ModelConfig) -> None:
        """Upload output model to cloud cache.

            model path and adapter path (if exists)
                will be uploaded to cloud cache to `<output_model_hash>/model_path.json`.
            model files will be uploaded to cloud cache to `<output_model_hash>/model/`.
            adapter files will be uploaded to cloud cache to `<output_model_hash>/adapter/`.
            HF model with model path as repo name will not be uploaded to cloud cache.

        Args:
            output_model_hash (str): Output model hash
            output_model_config (ModelConfig): Output model config

        """
        logger.info("Uploading model %s to cloud cache ...", output_model_hash)
        model_path = output_model_config.config.get("model_path")

        if model_path is None:
            logger.error("Model path is not found in the output model config. Upload failed.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path_map = {}
            adapter_path = None
            if output_model_config.type.lower() == "hfmodel" and is_hf_repo_exist(model_path):
                model_path_map["model_path"] = model_path
            else:
                model_path_map["model_path"] = Path(model_path).name

            if output_model_config.type.lower() == "hfmodel" and output_model_config.config.get("adapter_path"):
                adapter_path = Path(output_model_config.config["adapter_path"])
                model_path_map["adapter_path"] = adapter_path.name

            model_config_copy = deepcopy(output_model_config)

            model_config_copy.config["model_path"] = None
            if output_model_config.type.lower() == "hfmodel":
                model_config_copy.config["adapter_path"] = None

            # upload model file
            model_blob = str(Path(output_model_hash) / f"model/{model_path_map['model_path']}")
            adapter_blob = str(Path(output_model_hash) / "adapter")

            self.upload_model_files(model_path, model_blob)
            self.upload_model_files(adapter_path, adapter_blob)

            model_path_file = Path(temp_dir) / self.model_path_map
            with open(model_path_file, "w") as file:
                file.write(json.dumps(model_path_map))
            self._upload_file_to_blob(model_path_file, f"{output_model_hash}/{self.model_path_map}")

        # upload model config file
        model_config_bytes = json.dumps(model_config_copy.to_json()).encode()
        with io.BytesIO(model_config_bytes) as data:
            self.container_client.upload_blob(f"{output_model_hash}/model_config.json", data=data, overwrite=False)

    def upload_model_files(self, model_path: str, model_blob: str):
        if model_path:
            model_path = Path(model_path)
            # if HF model, model_path is a repo name, no need to upload
            if model_path.exists():
                if not model_path.is_dir():
                    self._upload_file_to_blob(model_path, model_blob)
                else:
                    self._upload_dir_to_blob(model_path, model_blob)

    def _upload_dir_to_blob(self, dir_path: Path, blob_folder_name: str):
        for item in dir_path.iterdir():
            if item.is_dir():
                self._upload_dir_to_blob(item, f"{blob_folder_name}/{item.name}")
            else:
                blob_name = f"{blob_folder_name}/{item.name}"
                self._upload_file_to_blob(item, blob_name)

    def _upload_file_to_blob(self, file_path: Path, blob_name: str):
        logger.info("Uploading %s to %s", file_path, blob_name)
        with open(file_path, "rb") as data:
            self.container_client.upload_blob(name=blob_name, data=data, overwrite=False)

    def _download_blob_list(
        self, blob_list, directory_prefix: str, output_model_path: Path, prefix: str = None
    ) -> None:
        for blob in blob_list:
            blob_client = self.container_client.get_blob_client(blob)
            local_file_path = (
                output_model_path / prefix / blob.name[len(directory_prefix) + 1 :]
                if prefix
                else output_model_path / blob.name[len(directory_prefix) + 1 :]
            )
            logger.info("Downloading %s to %s", blob.name, local_file_path)
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())


def check_model_cache(
    cloud_cache_helper: CloudCacheHelper, output_model_hash: str, output_model_path: Path
) -> Optional[ModelConfig]:
    """Check model cache in cloud cache.

    If model is found in cloud cache, download model config file and return model config.

    """
    output_model_config = None

    if cloud_cache_helper.exist_in_cloud_cache(output_model_hash):
        logger.info("Model is found in cloud cache.")
        output_model_config = cloud_cache_helper.get_model_config_by_hash_key(output_model_hash, output_model_path)
    else:
        logger.info("Model is not found in cloud cache.")
    return output_model_config


def update_input_model_config(
    cloud_cache_helper: CloudCacheHelper,
    input_model_config: ModelConfig,
    input_model_hash: str,
    output_model_path: Path,
) -> None:
    """Update input model config with model path and adapter path from cloud cache.

    Args:
        cloud_cache_helper (CloudCacheHelper): Cloud cache helper
        input_model_config (ModelConfig): Input model config
        input_model_hash (str): Input model hash
        output_model_path (Path): Output model path. Model files will be downloaded to this path.

    """
    # download model files
    logger.info("Cloud model cache is enabled. Downloading input model files ...")
    if cloud_cache_helper.exist_in_cloud_cache(input_model_hash):
        model_path_blob = f"{input_model_hash}/model_path.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path_file = Path(temp_dir) / "model_path.json"
            blob_client = cloud_cache_helper.container_client.get_blob_client(model_path_blob)
            with open(model_path_file, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

            with open(model_path_file) as file:
                model_path_map = json.load(file)
                model_path = model_path_map["model_path"]
                adapter_path = model_path_map.get("adapter_path")
                cloud_cache_helper.update_model_config(
                    model_path, adapter_path, input_model_config, input_model_hash, output_model_path
                )


def is_hf_repo_exist(repo_name: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.exception(
            "huggingface_hub is not installed. "
            "Please install huggingface_hub to use the cloud model cache feature for Huggingface model."
        )
        raise

    return HfApi().repo_exists(repo_name)
