# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from typing import Dict

from pydantic import validator

from olive.common.config_utils import ConfigBase

logger = logging.getLogger(__name__)


class AzureMLClientConfig(ConfigBase):
    subscription_id: str = None
    resource_group: str = None
    workspace_name: str = None
    aml_config_path: str = None
    # read timeout in seconds for HTTP requests, user can increase if they find the default value too small.
    # The default value from azureml sdk is 3000 which is too large and cause the evaluations and pass runs to
    # sometimes hang for a long time between retries of job stream and download steps.
    read_timeout: int = 60

    @validator("aml_config_path", always=True)
    def validate_aml_config_path(cls, v, values):
        if v is None:
            if values.get("subscription_id") is None:
                raise ValueError("subscription_id must be provided if aml_config_path is not provided")
            if values.get("resource_group") is None:
                raise ValueError("resource_group must be provided if aml_config_path is not provided")
            if values.get("workspace_name") is None:
                raise ValueError("workspace_name must be provided if aml_config_path is not provided")
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
            with open(self.aml_config_path, "r") as f:
                return json.load(f)
        else:
            # If aml_config_path is not provided, return the config from the class.
            return {
                "subscription_id": self.subscription_id,
                "resource_group": self.resource_group,
                "workspace_name": self.workspace_name,
            }

    def create_client(self):
        return AzureMLClient(**self.dict())


class AzureMLClient(object):
    def __init__(
        self,
        subscription_id: str = None,
        resource_group: str = None,
        workspace_name: str = None,
        aml_config_path: str = None,
        read_timeout: int = 60,
    ):
        from azure.ai.ml import MLClient

        # set logger level to error to avoid too many logs from azure-ai-ml sdk
        logging.getLogger("azure.ai.ml").setLevel(logging.ERROR)

        logger.info("Please make sure you have logged in Azure CLI and set the default subscription.")
        try:
            if aml_config_path is None:
                self.ml_client = MLClient(
                    credential=self._get_credentials(),
                    subscription_id=subscription_id,
                    resource_group_name=resource_group,
                    workspace_name=workspace_name,
                    read_timeout=read_timeout,
                )
            else:
                self.ml_client = MLClient.from_config(
                    credential=self._get_credentials(), path=aml_config_path, read_timeout=read_timeout
                )
        except Exception as e:
            logger.error(f"Failed to create AzureMLClient. Error: {e}")
            raise e

    def _get_credentials(self):
        from azure.identity import AzureCliCredential, DefaultAzureCredential, InteractiveBrowserCredential

        try:
            credential = AzureCliCredential()
            credential.get_token("https://management.azure.com/.default")
        except Exception:
            try:
                credential = DefaultAzureCredential()
                # Check if given credential can get token successfully.
                credential.get_token("https://management.azure.com/.default")
            except Exception:
                # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
                credential = InteractiveBrowserCredential()

        return credential
