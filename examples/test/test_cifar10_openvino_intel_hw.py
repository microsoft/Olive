# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest
from utils import check_output


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "cifar10_openvino_intel_hw"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


def test_cifar10():
    from olive.workflows import run as olive_run
    
    footprint = olive_run("config.json")
    check_output(footprint)
