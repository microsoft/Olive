# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

from olive.cli.base import BaseOliveCLICommand
from olive.common.utils import get_credentials


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
            required=True,
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
            self._delete_model_cache(client, self.args.model_hash)

    def _delete_model_cache(self, client, model_hash):
        for blob in client.list_blobs(model_hash):
            print("Deleting %s", blob.name)
            client.delete_blob(blob.name)

        if any(client.list_blobs(model_hash)):
            print("Deletion of the model cache with hash %s failed. Please try again.", model_hash)
        else:
            print("Model cache with hash %s removed from the cloud cache successfully.", model_hash)
