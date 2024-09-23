# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from argparse import ArgumentParser
from pathlib import Path

from olive.cli.base import BaseOliveCLICommand

logger = logging.getLogger(__name__)


class ManageAMLComputeCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser("manage-aml-compute", help="Create new compute in your AzureML workspace")
        group = sub_parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--create", "-c", action="store_true", help="Create new compute")
        group.add_argument("--delete", "-d", action="store_true", help="Delete existing compute")
        sub_parser.add_argument("--subscription_id", type=str, required=False, help="Azure subscription ID")
        sub_parser.add_argument("--resource_group", type=str, required=False, help="Name of the Azure resource group")
        sub_parser.add_argument("--workspace_name", type=str, required=False, help="Name of the AzureML workspace")
        sub_parser.add_argument(
            "--aml_config_path",
            type=str,
            required=False,
            help=(
                "Path to AzureML config file. If provided, subscription_id, resource_group and workspace_name are"
                " ignored"
            ),
        )
        sub_parser.add_argument("--compute_name", type=str, required=True, help="Name of the new compute")
        sub_parser.add_argument(
            "--vm_size",
            type=str,
            required=False,
            help="VM size of the new compute. This is required if you are creating a compute instance",
        )
        sub_parser.add_argument(
            "--location",
            type=str,
            required=False,
            help="Location of the new compute. This is required if you are creating a compute instance",
        )
        sub_parser.add_argument("--min_nodes", type=int, required=False, default=0, help="Minimum number of nodes")
        sub_parser.add_argument("--max_nodes", type=int, required=False, default=2, help="Maximum number of nodes")
        sub_parser.add_argument(
            "--idle_time_before_scale_down", type=int, required=False, default=120, help="Idle seconds before scaledown"
        )
        sub_parser.set_defaults(func=ManageAMLComputeCommand)

    def run(self):
        try:
            from azure.ai.ml.entities import AmlCompute
        except ImportError:
            raise ImportError(
                "azure-ai-ml is not installed. Please install azure-ai-ml packages to use this script."
            ) from None

        print("Running create new compute script...")

        ml_client = self.get_ml_client(
            self.args.aml_config_path, self.args.subscription_id, self.args.resource_group, self.args.workspace_name
        )

        if self.args.create:
            print(f"Creating compute {self.args.compute_name}...")
            if self.args.vm_size is None:
                raise ValueError("vm_size must be provided if operation is create")
            if self.args.location is None:
                raise ValueError("location must be provided if operation is create")
            cluster_basic = AmlCompute(
                name=self.args.compute_name,
                type="amlcompute",
                size=self.args.vm_size,
                location=self.args.location,
                min_instances=self.args.min_nodes,
                max_instances=self.args.max_nodes,
                idle_time_before_scale_down=self.args.idle_time_before_scale_down,
            )
            ml_client.begin_create_or_update(cluster_basic).result()
            print(
                f"Successfully created compute: {self.args.compute_name} at {self.args.location} "
                f"with vm_size:{self.args.vm_size} and min_nodes={self.args.min_nodes} and "
                f"max_nodes={self.args.max_nodes} and "
                f"idle_time_before_scale_down={self.args.idle_time_before_scale_down}"
            )
        elif self.args.delete:
            print(f"Deleting compute {self.args.compute_name}...")
            ml_client.compute.begin_delete(self.args.compute_name).wait()
            print(f"Successfully deleted compute: {self.args.compute_name}")

    @classmethod
    def get_ml_client(cls, aml_config_path: str, subscription_id: str, resource_group: str, workspace_name: str):
        try:
            from azure.ai.ml import MLClient
        except ImportError:
            raise ImportError(
                "azure-ai-ml is not installed. Please install azure-ai-ml packages to use this script."
            ) from None

        if aml_config_path is not None:
            if not Path(aml_config_path).exists():
                raise ValueError(f"aml_config_path {aml_config_path} does not exist")
            if not Path(aml_config_path).is_file():
                raise ValueError(f"aml_config_path {aml_config_path} is not a file")
            return MLClient.from_config(credential=cls.get_credentials(), path=aml_config_path)
        else:
            if subscription_id is None:
                raise ValueError("subscription_id must be provided if aml_config_path is not provided")
            if resource_group is None:
                raise ValueError("resource_group must be provided if aml_config_path is not provided")
            if workspace_name is None:
                raise ValueError("workspace_name must be provided if aml_config_path is not provided")
            return MLClient(
                credential=cls.get_credentials(),
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name,
            )

    @staticmethod
    def get_credentials():
        try:
            from azure.identity import AzureCliCredential, DefaultAzureCredential, InteractiveBrowserCredential
        except ImportError:
            raise ImportError(
                "azure-identity is not installed. Please install azure-identity packages to use this command."
            ) from None

        print("Getting credentials for MLClient")
        try:
            credential = AzureCliCredential()
            credential.get_token("https://management.azure.com/.default")
            print("Using AzureCliCredential")
        except Exception:
            try:
                credential = DefaultAzureCredential()
                # Check if given credential can get token successfully.
                credential.get_token("https://management.azure.com/.default")
                print("Using DefaultAzureCredential")
            except Exception:
                # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
                credential = InteractiveBrowserCredential()
                print("Using InteractiveBrowserCredential")

        return credential
