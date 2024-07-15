# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

# pylint: disable=broad-exception-raised


def get_olive_workspace_config():
    subscription_id = os.environ.get("WORKSPACE_SUBSCRIPTION_ID")
    if subscription_id is None:
        raise Exception("Please set the environment variable WORKSPACE_SUBSCRIPTION_ID")

    resource_group = os.environ.get("WORKSPACE_RESOURCE_GROUP")
    if resource_group is None:
        raise Exception("Please set the environment variable WORKSPACE_RESOURCE_GROUP")

    workspace_name = os.environ.get("WORKSPACE_NAME")
    if workspace_name is None:
        raise Exception("Please set the environment variable WORKSPACE_NAME")

    exclude_managed_identity_credential = (
        {"exclude_managed_identity_credential": True} if "EXCLUDE_MANAGED_IDENTITY_CREDENTIAL" in os.environ else {}
    )

    client_id = os.environ.get("MANAGED_IDENTITY_CLIENT_ID")
    if client_id is None and not exclude_managed_identity_credential:
        raise Exception("Please set the environment variable MANAGED_IDENTITY_CLIENT_ID")

    return {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
        # pipeline agents have multiple managed identities, so we need to specify the client_id
        "default_auth_params": {"managed_identity_client_id": client_id, **exclude_managed_identity_credential},
    }


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
