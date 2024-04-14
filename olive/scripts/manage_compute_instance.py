# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import logging
from pathlib import Path

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import AmlCompute
except ImportError:
    raise ImportError("azure-ai-ml is not installed. Please install azure-ai-ml packages to use this script.") from None

try:
    from azure.identity import AzureCliCredential, DefaultAzureCredential, InteractiveBrowserCredential
except ImportError:
    raise ImportError(
        "azure-identity is not installed. Please install azure-identity packages to use this script."
    ) from None

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Create new compute in your AzureML workspace")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", "-c", action="store_true", help="Create new compute")
    group.add_argument("--delete", "-d", action="store_true", help="Delete existing compute")
    parser.add_argument("--subscription_id", type=str, required=False, help="Azure subscription ID")
    parser.add_argument("--resource_group", type=str, required=False, help="Name of the Azure resource group")
    parser.add_argument("--workspace_name", type=str, required=False, help="Name of the AzureML workspace")
    parser.add_argument(
        "--aml_config_path",
        type=str,
        required=False,
        help="Path to AzureML config file. If provided, subscription_id, resource_group and workspace_name are ignored",
    )
    parser.add_argument("--compute_name", type=str, required=True, help="Name of the new compute")
    parser.add_argument(
        "--vm_size",
        type=str,
        required=False,
        help="VM size of the new compute. This is required if you are creating a compute instance",
    )
    parser.add_argument(
        "--location",
        type=str,
        required=False,
        help="Location of the new compute. This is required if you are creating a compute instance",
    )
    parser.add_argument("--min_nodes", type=int, required=False, default=0, help="Minimum number of nodes")
    parser.add_argument("--max_nodes", type=int, required=False, default=2, help="Maximum number of nodes")
    parser.add_argument(
        "--idle_time_before_scale_down", type=int, required=False, default=120, help="Idle seconds before scaledown"
    )
    return parser.parse_args()


def main():
    logger.info("Running create new compute script...")
    args = get_args()
    aml_config_path = args.aml_config_path
    subscription_id = args.subscription_id
    resource_group = args.resource_group
    workspace_name = args.workspace_name
    ml_client = get_ml_client(aml_config_path, subscription_id, resource_group, workspace_name)
    compute_name = args.compute_name

    is_create = args.create
    is_delete = args.delete
    if is_create:
        logger.info("Creating compute %s...", compute_name)
        vm_size = args.vm_size
        location = args.location
        min_nodes = args.min_nodes
        max_nodes = args.max_nodes
        idle_time_before_scale_down = args.idle_time_before_scale_down
        if vm_size is None:
            raise ValueError("vm_size must be provided if operation is create")
        if location is None:
            raise ValueError("location must be provided if operation is create")
        cluster_basic = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size=vm_size,
            location=location,
            min_instances=min_nodes,
            max_instances=max_nodes,
            idle_time_before_scale_down=idle_time_before_scale_down,
        )
        ml_client.begin_create_or_update(cluster_basic).result()
        logger.info(
            "Successfully created compute: %s at %s with vm_size:%s and "
            "min_nodes=%d and max_nodes=%d and idle_time_before_scale_down=%d",
            compute_name,
            location,
            vm_size,
            min_nodes,
            max_nodes,
            idle_time_before_scale_down,
        )
    elif is_delete:
        logger.info("Deleting compute %s...", compute_name)
        ml_client.compute.begin_delete(compute_name).wait()
        logger.info("Successfully deleted compute: %s", compute_name)


def get_ml_client(aml_config_path, subscription_id, resource_group, workspace_name):
    if aml_config_path is not None:
        if not Path(aml_config_path).exists():
            raise ValueError(f"aml_config_path {aml_config_path} does not exist")
        if not Path(aml_config_path).is_file():
            raise ValueError(f"aml_config_path {aml_config_path} is not a file")
        return MLClient.from_config(credential=get_credentials(), path=aml_config_path)
    else:
        if subscription_id is None:
            raise ValueError("subscription_id must be provided if aml_config_path is not provided")
        if resource_group is None:
            raise ValueError("resource_group must be provided if aml_config_path is not provided")
        if workspace_name is None:
            raise ValueError("workspace_name must be provided if aml_config_path is not provided")
        return MLClient(
            credential=get_credentials(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )


def get_credentials():
    logger.info("Getting credentials for MLClient")
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
        logger.info("Using AzureCliCredential")
    except Exception:
        try:
            credential = DefaultAzureCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
            logger.info("Using DefaultAzureCredential")
        except Exception:
            # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
            credential = InteractiveBrowserCredential()
            logger.info("Using InteractiveBrowserCredential")

    return credential


if __name__ == "__main__":
    main()
