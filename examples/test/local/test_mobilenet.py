# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest

from olive.common.hf.login import huggingface_login

from ..utils import check_output, get_example_dir, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("mobilenet/onnx"))


def test_mobilenet():
    from olive.workflows import run as olive_run

    hf_token = os.environ.get("HF_TOKEN")
    huggingface_login(hf_token)

    olive_config = patch_config("config.json")
    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
