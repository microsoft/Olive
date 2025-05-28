# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os

import pytest

from olive.common.hf.login import huggingface_login

from ..utils import assert_nodes, get_example_dir


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("phi2"))


def test_phi2_genai():
    from olive.workflows import run as olive_run

    hf_token = os.environ.get("HF_TOKEN")
    huggingface_login(hf_token)

    with open("phi2_genai.json") as f:
        olive_config = json.load(f)

    workflow_output = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    assert_nodes(workflow_output)
