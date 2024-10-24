import logging

from olive.common.constants import ACCOUNT_URL_TEMPLATE
from olive.common.utils import get_credentials

logger = logging.getLogger(__name__)


class AzureContainerClientFactory:
    def __init__(self, account_name, container_name, **credential_kwargs):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as exc:
            raise ImportError(
                "Please install azure-storage-blob and azure-identity to use the shared cache feature."
            ) from exc

        self.client = ContainerClient(
            account_url=ACCOUNT_URL_TEMPLATE.format(account_name=account_name),
            container_name=container_name,
            credential=get_credentials(**credential_kwargs),
        )

    def delete_blob(self, blob_name):
        for blob in self.client.list_blobs(blob_name):
            logger.info("Deleting %s", blob.name)
            self.client.delete_blob(blob.name)

        if self.exists(blob_name):
            logger.error("Deletion of the files %s failed. Please try again.", blob_name)
        else:
            logger.info("Files %s removed from the shared cache successfully.", blob_name)

    def delete_all(self):
        for blob in self.client.list_blobs():
            logger.info("Deleting %s", blob.name)
            self.client.delete_blob(blob.name)

        if self.exists():
            logger.error("Deletion of all files failed. Please try again.")
        else:
            logger.info("All files removed from the shared cache successfully.")

    def upload_blob(self, blob_name, data, overwrite=False):
        self.client.upload_blob(blob_name, data=data, overwrite=overwrite)
        logger.info("File %s uploaded to the shared cache successfully.", blob_name)

    def download_blob(self, blob_name, file_path):
        blob_client = self.client.get_blob_client(blob_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())
        logger.debug("File %s downloaded to %s successfully.", blob_name, file_path)

    def exists(self, blob_name=None):
        return any(self.client.list_blobs(blob_name))
