# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from olive.common.utils import hash_dict
from olive.engine.cloud_cache_helper import CloudCacheHelper, CloudModelInvalidStatus
from olive.model.config.model_config import ModelConfig


class TestCloudCacheHelper:
    # pylint: disable=W0201

    @pytest.fixture(autouse=True)
    @patch("olive.common.utils.get_credentials")
    @patch("azure.storage.blob.ContainerClient")
    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"model": "model_path"}).encode("utf-8"))
    def setup(self, mock_file, mock_container_client, mock_get_credentials):
        cache_dir = "cache_dir"
        account_url = "account_url"
        container_name = "container_name"
        hf_cache_path = "hf_cache_path"
        mock_blob_client = MagicMock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = "exist"
        mock_download_stream = MagicMock()
        mock_blob_client.download_blob.return_value = mock_download_stream
        mock_download_stream.readall.return_value = json.dumps({"model": "model_path"}, indent=2).encode("utf-8")

        self.cloud_cache_helper = CloudCacheHelper(cache_dir, account_url, container_name, hf_cache_path)

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

    def test_download_model_cache_map_not_exist(self):
        # setup
        mock_blob_client = MagicMock()
        self.cloud_cache_helper.container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = None

        # execute
        model_cache_map = self.cloud_cache_helper.download_model_cache_map()

        # assert
        assert model_cache_map == {}

    def test_download_model_cache_map(self):
        # setup
        expected_model_cache_map = {"model": "map"}
        mock_blob_client = MagicMock()
        self.cloud_cache_helper.container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.exists.return_value = "exist"
        mock_download_stream = MagicMock()
        mock_blob_client.download_blob.return_value = mock_download_stream
        mock_download_stream.readall.return_value = json.dumps(expected_model_cache_map, indent=2).encode("utf-8")

        # execute
        actual_model_cache_map = self.cloud_cache_helper.download_model_cache_map()

        # assert
        assert actual_model_cache_map == expected_model_cache_map

    def test_get_hash_key_with_input_hash(self):
        # setup
        model_config = ModelConfig(type="onnxmodel", config={"model_path": "model.onnx"})
        expected_model_config = ModelConfig(type="onnxmodel", config={"model_path": None})
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

    def test_exist_in_model_cache_map(self):
        # setup
        output_model_hash = "model"
        expected_model_path = "model_path"

        # execute
        actual_model_path = self.cloud_cache_helper.exist_in_model_cache_map(output_model_hash)

        # assert
        assert expected_model_path == actual_model_path

    def test_exist_in_model_cache_map_not_found(self):
        # setup
        output_model_hash = "hash"

        # execute
        actual_model_path = self.cloud_cache_helper.exist_in_model_cache_map(output_model_hash)

        # assert
        assert actual_model_path is None

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

    def test_upload_model_to_cloud_cache_no_model(self):
        # setup
        output_model_config = ModelConfig(type="onnxmodel", config={})

        # execute
        actual_return = self.cloud_cache_helper.upload_model_to_cloud_cache("hash", output_model_config)

        # assert
        assert actual_return == str(CloudModelInvalidStatus.NO_MODEL_FILE)

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
            actual_return = self.cloud_cache_helper.upload_model_to_cloud_cache(output_model_hash, output_model_config)

            assert self.cloud_cache_helper.container_client.upload_blob.call_count == 2
            assert actual_return == Path("model.onnx").name
