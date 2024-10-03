# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.cli.base import BaseOliveCLICommand
from olive.common.utils import get_credentials

logger = logging.getLogger(__name__)


class CloudCacheCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser):
        sub_parser = parser.add_parser("cloud-cache", help="Cloud cache model operations")
        sub_parser.add_argument(
            "--delete",
            action="store_true",
            help="Delete a model cache from the cloud cache.",
        )
        sub_parser.add_argument(
            "--all",
            action="store_true",
            help="Delete all model cache from the cloud cache.",
        )
        sub_parser.add_argument(
            "--account",
            type=str,
            required=True,
            help="The account name for the cloud cache.",
        )
        sub_parser.add_argument(
            "--container",
            type=str,
            required=True,
            help="The container name for the cloud cache.",
        )
        sub_parser.add_argument(
            "--model_hash",
            type=str,
            required=False,
            help="The model hash to remove from the cloud cache.",
        )
        sub_parser.set_defaults(func=CloudCacheCommand)

    def run(self):
        try:
            from azure.storage.blob import ContainerClient
        except ImportError as exc:
            raise ImportError(
                "Please install azure-storage-blob and azure-identity to use the cloud model cache feature."
            ) from exc

        account_url = f"https://{self.args.account}.blob.core.windows.net"
        client = ContainerClient(
            account_url=account_url,
            container_name=self.args.container,
            credential=get_credentials({"exclude_managed_identity_credential": True}),
        )

        if self.args.delete:
            if self.args.all:
                self._delete_all_model_cache(client)
            else:
                if not self.args.model_hash:
                    logger.error("Please provide a model hash to delete.")
                    return
                self._delete_model_cache(client, self.args.model_hash)     
                
    def _delete_all_model_cache(self, client):
        for blob in client.list_blobs():
            logger.info("Deleting %s", blob.name)
            client.delete_blob(blob.name)

        if any(client.list_blobs()):
            logger.error("Deletion of all model cache failed. Please try again.")
        else:
            logger.info("All model cache removed from the cloud cache successfully.")

    def _delete_model_cache(self, client, model_hash):
        for blob in client.list_blobs(model_hash):
            logger.info("Deleting %s", blob.name)
            client.delete_blob(blob.name)

        if any(client.list_blobs(model_hash)):
            logger.error("Deletion of the model cache with hash %s failed. Please try again.", model_hash)
        else:
            logger.info("Model cache with hash %s removed from the cloud cache successfully.", model_hash)
