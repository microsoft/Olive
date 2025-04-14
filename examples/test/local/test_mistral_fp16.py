# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os

import pytest

from olive.common.hf.login import huggingface_login

from ..utils import check_output, get_example_dir


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("mistral"))


@pytest.mark.parametrize("olive_json", ["mistral_fp16.json"])
def test_mistral(olive_json):
    from olive.workflows import run as olive_run

    hf_token = os.environ.get("HF_TOKEN")
    huggingface_login(hf_token)

    with open(olive_json) as f:
        olive_config = json.load(f)

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
