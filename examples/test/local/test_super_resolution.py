# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os

import pytest

from ..utils import assert_nodes, get_example_dir


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("super_resolution"))


def test_super_resolution():
    from olive.workflows import run as olive_run

    with open("config.json") as f:
        olive_config = json.load(f)

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    assert_nodes(footprint)
