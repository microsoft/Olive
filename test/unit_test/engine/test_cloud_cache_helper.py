# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import shutil
from unittest.mock import MagicMock, mock_open, patch

from olive.common.utils import hash_dict
from olive.engine.cloud_cache_helper import CloudCacheHelper
from olive.model.config.model_config import ModelConfig


class TestCloudCacheHelper:
    # pylint: disable=W0201

    @patch("olive.common.utils.get_credentials")
    @patch("azure.storage.blob.ContainerClient")
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"model": "model_path"}).encode("utf-8"))
    def setup_method(self, method, mock_file, mock_container_client, mock_get_credentials):
        self.cache_dir = "cache_dir"
        account_url = "account_url"
        container_name = "container_name"
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = "exist"
        mock_download_stream = MagicMock()
        mock_blob_client.download_blob.return_value = mock_download_stream
        mock_download_stream.readall.return_value = json.dumps({"model": "model_path"}, indent=2).encode("utf-8")

        self.cloud_cache_helper = CloudCacheHelper(self.cache_dir, account_url, container_name)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("cache_dir")

    def test_update_model_config(self):
        # setup
        input_model_hash = "input_model_hash"
        mock_blob = MagicMock(name=f"{input_model_hash}/model/blob")
        self.cloud_cache_helper.container_client.list_blobs.return_value = mock_blob
        mock_blob_client = MagicMock()
        self.cloud_cache_helper.container_client.get_blob_client.return_value = mock_blob_client
        mock_download_stream = MagicMock()
        mock_blob_client.download_blob.return_value = mock_download_stream
        mock_download_stream.readall.return_value = "model"
        cloud_model_path = "cloud_model_path"
        model_config = ModelConfig(type="onnxmodel", config={})

        # execute
        self.cloud_cache_helper.update_model_config(cloud_model_path, model_config, input_model_hash)

        # assert
        assert model_config.config["model_path"] == str(self.cloud_cache_helper.output_model_path / cloud_model_path)

    def test_get_hash_key_with_input_hash(self):
        # setup
        model_config = ModelConfig(type="onnxmodel", config={"model_path": "model.onnx"})
        expected_model_config = ModelConfig(type="onnxmodel", config={})
        pass_search_point = {"search_point": "point"}
        input_model_hash = "input_model_hash"
        expected_hash_key = hash_dict(
            {
                "hf_hub_model_commit_id": None,
                "model_config": expected_model_config.to_json(),
                "pass_search_point": pass_search_point,
                "input_model_hash": input_model_hash,
            }
        )

        # execute
        actual_hash_key = self.cloud_cache_helper.get_hash_key(model_config, pass_search_point, input_model_hash)

        # assert
        assert expected_hash_key == actual_hash_key

    @patch("huggingface_hub.repo_info")
    def test_get_hash_key_without_input_hash(self, mock_repo_info):
        # setup
        model_config = ModelConfig(type="HfModel", config={"model_path": "model_name"})
        pass_search_point = {"search_point": "point"}
        input_model_hash = None
        mock_repo_info.return_value.sha = "sha"
        expected_hash_key = hash_dict(
            {
                "hf_hub_model_commit_id": "sha",
                "model_config": model_config.to_json(),
                "pass_search_point": pass_search_point,
                "input_model_hash": input_model_hash,
            }
        )

        # execute
        actual_hash_key = self.cloud_cache_helper.get_hash_key(model_config, pass_search_point, input_model_hash)

        # assert
        assert expected_hash_key == actual_hash_key

    def test_get_model_config_by_hash_key(self):
        # setup
        output_model_hash = "output_model_hash"
        mock_blob_client = MagicMock()
        self.cloud_cache_helper.container_client.get_blob_client.return_value = mock_blob_client
        mock_download_stream = MagicMock()
        mock_blob_client.download_blob.return_value = mock_download_stream
        expected_model_config = ModelConfig(type="onnxmodel", config={})
        model_config_json = expected_model_config.to_json()
        mock_download_stream.readall.return_value = json.dumps(model_config_json, indent=2).encode("utf-8")

        # execute
        actual_model_config = self.cloud_cache_helper.get_model_config_by_hash_key(output_model_hash)

        # assert
        assert expected_model_config == actual_model_config

    def test_upload_model_to_cloud_cache(self):
        # setup
        output_model_hash = "output_model_hash"
        output_model_config = ModelConfig(type="onnxmodel", config={"model_path": "model.onnx"})
        model_binary_data = b"Test binary data"
        m = mock_open(read_data=model_binary_data)
        with patch("builtins.open", m):
            # Your function or code that uses open
            with open("model.onnx", "rb") as model_data:
                model_data.read()

            # execute
            self.cloud_cache_helper.upload_model_to_cloud_cache(output_model_hash, output_model_config)

            # assert
            # 1. model_config.json 2. model 3. model_path.json
            assert self.cloud_cache_helper.container_client.upload_blob.call_count == 3
