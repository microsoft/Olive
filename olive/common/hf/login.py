# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

logger = logging.getLogger(__name__)


def huggingface_login(token: str):
    from huggingface_hub import login

    login(token=token)
