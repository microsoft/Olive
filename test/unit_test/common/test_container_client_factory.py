from unittest.mock import MagicMock, call, patch

import pytest

from olive.common.container_client_factory import AzureContainerClientFactory


class TestAzureContainerClientFactory:
    @pytest.fixture(autouse=True)
    @patch("olive.common.utils.get_credentials")
    @patch("azure.storage.blob.ContainerClient")
    def setup(self, mock_ContainerClient, mock_get_credentials):
        self.container_client = AzureContainerClientFactory("dummy_account", "dummy_container")
        self.mock_ContainerClient = mock_ContainerClient

    def test_delete_blob(self):
        # setup
        mock_blob = MagicMock()
        mock_blob.name = "dummy_blob"
        self.container_client.client.list_blobs.return_value = [mock_blob]

        # execute
        self.container_client.delete_blob("dummy_blob")

        # assert
        self.container_client.client.delete_blob.assert_called_once_with("dummy_blob")

    def test_delete_all(self):
        # setup
        mock_blob = MagicMock()
        mock_blob.name = "dummy_blob"
        mock_blob2 = MagicMock()
        mock_blob2.name = "dummy_blob2"
        self.container_client.client.list_blobs.return_value = [mock_blob, mock_blob2]

        # execute
        self.container_client.delete_all()

        # assert
        self.container_client.client.delete_blob.assert_has_calls([call("dummy_blob"), call("dummy_blob2")])
        assert self.container_client.client.delete_blob.call_count == 2

    def test_upload_blob(self):
        # setup
        data = b"dummy_data"

        # execute
        self.container_client.upload_blob("dummy_blob", data, overwrite=True)

        # assert
        self.container_client.client.upload_blob.assert_called_once_with("dummy_blob", data=data, overwrite=True)

    def test_download_blob(self, tmp_path):
        # setup
        blob_name = "dummy_blob"
        file_path = tmp_path / "downloaded_blob"
        blob_client_mock = MagicMock()
        self.container_client.client.get_blob_client.return_value = blob_client_mock
        blob_client_mock.download_blob.return_value.readall.return_value = b"dummy_data"

        # execute
        self.container_client.downlaod_blob(blob_name, file_path)

        # assert
        blob_client_mock.download_blob.assert_called_once()
        with open(file_path, "rb") as f:
            assert f.read() == b"dummy_data"

    def test_exists(self):
        # setup
        self.container_client.client.list_blobs.return_value = [MagicMock()]

        # assert
        assert self.container_client.exists("dummy_blob") is True
        self.container_client.client.list_blobs.assert_called_once_with("dummy_blob")

    def test_exists_not_found(self):
        # setup
        self.container_client.client.list_blobs.return_value = []

        # assert
        assert self.container_client.exists("dummy_blob") is False
        self.container_client.client.list_blobs.assert_called_with("dummy_blob")
