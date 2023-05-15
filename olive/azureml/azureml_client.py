# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.common.config_utils import ConfigBase


logger = logging.getLogger(__name__)


class AzureMLClientConfig(ConfigBase):
    subscription_id: str
    resource_group: str
    workspace_name: str
    read_timeout: int = 60
    
    def create_client(self):
        return AzureMLClient(**self.dict())
    
class AzureMLClient(object):
    def __init__(
        self, 
        subscription_id: str = None, 
        resource_group: str = None, 
        workspace_name: str = None, 
        aml_config_path: str = None, 
        read_timeout: int = 60
    ):
        from azure.ai.ml import MLClient
        logger.info("Please make sure you have logged in Azure CLI and set the default subscription.")
        if aml_config_path is None:
            self.ml_client = MLClient(
                credential=self._get_credentials(), 
                subscription_id=subscription_id, 
                resource_group_name=resource_group, 
                workspace_name=workspace_name,
                read_timeout=read_timeout
            )
        else:
            self.ml_client = MLClient.from_config(credential=self._get_credentials(), path=aml_config_path, read_timeout=read_timeout)

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