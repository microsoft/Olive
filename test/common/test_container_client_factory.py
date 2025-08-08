from unittest.mock import MagicMock, call, patch

import pytest

from olive.common.container_client_factory import AzureContainerClientFactory

# pylint: disable=W0201, W0212


class TestAzureContainerClientFactory:
    @pytest.fixture(autouse=True)
    @patch("olive.common.utils.get_credentials")
    @patch("azure.storage.blob.ContainerClient")
    def setup(self, mock_ContainerClient, mock_get_credentials):
        self.container_client = AzureContainerClientFactory("dummy_account", "dummy_container")

    @patch("olive.common.container_client_factory.retry_func")
    def test_get_blob_list(self, mock_retry_func):
        # setup
        mock_blob = MagicMock()
        mock_blob.name = "dummy_blob"
        mock_retry_func.return_value = [mock_blob]

        # execute
        blob_list = self.container_client.get_blob_list("dummy_blob")

        # assert
        assert blob_list == [mock_blob]
        mock_retry_func.assert_called_once_with(self.container_client.client.list_blobs, ["dummy_blob"])

    @patch("olive.common.container_client_factory.retry_func")
    @patch("olive.common.container_client_factory.AzureContainerClientFactory.get_blob_list")
    def test_delete_blob(self, mock_get_blob_list, mock_retry_func):
        # setup
        mock_blob = MagicMock()
        mock_blob.name = "dummy_blob"
        mock_get_blob_list.return_value = [mock_blob]

        # execute
        self.container_client.delete_blob("dummy_blob")

        # assert
        mock_retry_func.assert_called_once_with(self.container_client.client.delete_blob, ["dummy_blob"])

    @patch("olive.common.container_client_factory.retry_func")
    @patch("olive.common.container_client_factory.AzureContainerClientFactory.get_blob_list")
    def test_delete_all(self, mock_get_blob_list, mock_retry_func):
        # setup
        mock_blob = MagicMock()
        mock_blob.name = "dummy_blob"
        mock_blob2 = MagicMock()
        mock_blob2.name = "dummy_blob2"
        mock_get_blob_list.return_value = [mock_blob, mock_blob2]

        # execute
        self.container_client.delete_all()

        # assert
        mock_retry_func.assert_has_calls(
            [
                call(self.container_client.client.delete_blob, ["dummy_blob"]),
                call(self.container_client.client.delete_blob, ["dummy_blob2"]),
            ]
        )
        assert mock_retry_func.call_count == 2

    @patch("olive.common.container_client_factory.retry_func")
    def test_upload_blob(self, mock_retry_func):
        # setup
        data = b"dummy_data"

        # execute
        self.container_client.upload_blob("dummy_blob", data, overwrite=True)

        # assert
        mock_retry_func.assert_called_once_with(
            self.container_client.client.upload_blob, ["dummy_blob", data], {"overwrite": True}
        )

    @patch("olive.common.container_client_factory.retry_func")
    @patch("olive.common.container_client_factory.AzureContainerClientFactory._download_blob")
    def test_download_blob(self, mock__download_blob, mock_retry_func):
        # execute
        self.container_client.download_blob("dummy_blob", "dummy_file_path")

        # assert
        mock_retry_func.assert_called_once_with(self.container_client._download_blob, ["dummy_blob", "dummy_file_path"])

    def test__download_blob(self, tmp_path):
        # setup
        blob_name = "dummy_blob"
        file_path = tmp_path / "downloaded_blob"
        blob_client_mock = MagicMock()
        self.container_client.client.get_blob_client.return_value = blob_client_mock
        blob_client_mock.download_blob.return_value.readall.return_value = b"dummy_data"

        # execute
        self.container_client._download_blob(blob_name, file_path)

        # assert
        blob_client_mock.download_blob.assert_called_once()
        with open(file_path, "rb") as f:
            assert f.read() == b"dummy_data"

    @pytest.mark.parametrize("mock_blob_list", [[MagicMock()], []])
    @patch("olive.common.container_client_factory.AzureContainerClientFactory.get_blob_list")
    def test_exists(self, mock_get_blob_list, mock_blob_list):
        # setup
        mock_get_blob_list.return_value = mock_blob_list

        # assert
        assert self.container_client.exists("dummy_blob") == bool(mock_blob_list)
