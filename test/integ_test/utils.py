# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

# pylint: disable=broad-exception-raised


def download_azure_blob(container, blob, download_path, storage_account="olivewheels"):
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobClient

    blob = BlobClient.from_blob_url(
        f"https://{storage_account}.blob.core.windows.net/{container}/{blob}",
        credential=ManagedIdentityCredential(client_id=os.environ.get("MANAGED_IDENTITY_CLIENT_ID")),
    )

    with open(download_path, "wb") as my_blob:
        blob_data = blob.download_blob()
        blob_data.readinto(my_blob)
