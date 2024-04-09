# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

from azure.storage.blob import BlobClient

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

    client_id = os.environ.get("MANAGED_IDENTITY_CLIENT_ID")
    if client_id is None:
        raise Exception("Please set the environment variable MANAGED_IDENTITY_CLIENT_ID")

    return {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
        # pipeline agents have multiple managed identities, so we need to specify the client_id
        "default_auth_params": {"managed_identity_client_id": client_id},
    }


def download_azure_blob(container, blob, download_path):
    try:
        conn_str = os.environ["OLIVEWHEELS_STORAGE_CONNECTION_STRING"]
    except KeyError as e:
        raise Exception("Please set the environment variable OLIVEWHEELS_STORAGE_CONNECTION_STRING") from e

    blob = BlobClient.from_connection_string(conn_str=conn_str, container_name=container, blob_name=blob)

    with open(download_path, "wb") as my_blob:
        blob_data = blob.download_blob()
        blob_data.readinto(my_blob)
