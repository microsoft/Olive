# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os

from olive.common.utils import get_credentials

logger = logging.getLogger(__name__)


def huggingface_login(token: str):
    from huggingface_hub import login

    login(token=token)


def aml_runner_hf_login():
    hf_login = os.environ.get("HF_LOGIN")
    if hf_login:
        from azure.keyvault.secrets import SecretClient

        keyvault_name = os.environ.get("KEYVAULT_NAME")
        logger.debug("Getting token from keyvault %s", keyvault_name)

        credential = get_credentials()
        secret_client = SecretClient(vault_url=f"https://{keyvault_name}.vault.azure.net/", credential=credential)
        token = secret_client.get_secret("hf-token").value
        huggingface_login(token)
