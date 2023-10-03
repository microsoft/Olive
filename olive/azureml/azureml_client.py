# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, validator

from olive.common.config_utils import ConfigBase

logger = logging.getLogger(__name__)


class AzureMLClientConfig(ConfigBase):
    subscription_id: str = Field(
        None, description="Azure subscription id. Required if aml_config_path is not provided."
    )
    resource_group: str = Field(None, description="Azure resource group. Required if aml_config_path is not provided.")
    workspace_name: str = Field(None, description="Azure workspace name. Required if aml_config_path is not provided.")
    aml_config_path: str = Field(
        None, description="Path to AzureML config file. If provided, other fields are ignored."
    )
    # read timeout in seconds for HTTP requests, user can increase if they find the default value too small.
    # The default value from azureml sdk is 3000 which is too large and cause the evaluations and pass runs to
    # sometimes hang for a long time between retries of job stream and download steps.
    read_timeout: int = Field(60, description="Read timeout in seconds for HTTP requests.")
    max_operation_retries: int = Field(
        3, description="Max number of retries for AzureML operations like resource creation or download."
    )
    operation_retry_interval: int = Field(
        5,
        description=(
            "Initial interval in seconds between retries for AzureML operations like resource creation or download. The"
            " interval doubles after each retry."
        ),
    )
    # as the DefaultAzureCredential is used by default, we need to provide the default auth config for it.
    # but DefaultAzureCredential accept kwargs as parameters, it is hard to validate the config.
    # so we just provide a dict here and let the user to provide the correct config following the doc.
    default_auth_params: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Default auth config for AzureML client. Please refer to"
            " https://learn.microsoft.com/en-us/python/api/azure-identity/"
            "azure.identity.defaultazurecredential?view=azure-python#parameters"
            " for more details."
        ),
    )

    @validator("aml_config_path", always=True)
    def validate_aml_config_path(cls, v, values):
        if v is not None:
            if not Path(v).exists():
                raise ValueError(f"aml_config_path {v} does not exist")
            if not Path(v).is_file():
                raise ValueError(f"aml_config_path {v} is not a file")
        return v

    def get_workspace_config(self) -> Dict[str, str]:
        """Get the workspace config as a dict."""
        if self.aml_config_path:
            # If aml_config_path is provided, load the config from the file.
            with open(self.aml_config_path) as f:  # noqa: PTH123
                return json.load(f)
        else:
            # If aml_config_path is not provided, return the config from the class.
            return {
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "workspace_name": self.workspace_name,
            }

    def create_client(self):
        """Create an MLClient instance."""
        from azure.ai.ml import MLClient

        # set logger level to error to avoid too many logs from azure sdk
        logging.getLogger("azure.ai.ml").setLevel(logging.ERROR)
        logging.getLogger("azure.identity").setLevel(logging.ERROR)

        if self.aml_config_path is None:
            if self.subscription_id is None:
                raise ValueError("subscription_id must be provided if aml_config_path is not provided")
            if self.resource_group is None:
                raise ValueError("resource_group must be provided if aml_config_path is not provided")
            if self.workspace_name is None:
                raise ValueError("workspace_name must be provided if aml_config_path is not provided")
            return MLClient(
                credential=self._get_credentials(),
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name,
                read_timeout=self.read_timeout,
            )
        else:
            return MLClient.from_config(
                credential=self._get_credentials(),
                path=self.aml_config_path,
                read_timeout=self.read_timeout,
            )

    def create_registry_client(self, registry_name: str):
        """Create an MLClient instance."""
        from azure.ai.ml import MLClient

        # set logger level to error to avoid too many logs from azure sdk
        logging.getLogger("azure.ai.ml").setLevel(logging.ERROR)
        logging.getLogger("azure.identity").setLevel(logging.ERROR)

        return MLClient(credential=self._get_credentials(), registry_name=registry_name)

    def _get_credentials(self):
        """Get credentials for MLClient.

        Order of credential providers:
        1. Azure CLI
        2. DefaultAzureCredential
        3. InteractiveBrowserCredential
        """
        from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

        logger.debug("Getting credentials for MLClient")
        try:
            default_auth_params = self.default_auth_params or {}
            credential = DefaultAzureCredential(**default_auth_params)
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
            logger.debug("Using DefaultAzureCredential")
        except Exception:
            # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
            credential = InteractiveBrowserCredential()
            logger.debug("Using InteractiveBrowserCredential")

        return credential
