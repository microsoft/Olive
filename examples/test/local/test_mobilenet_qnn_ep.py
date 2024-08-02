# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys

import pytest

from olive.hardware import AcceleratorSpec

from ..utils import get_example_dir


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    example_dir = get_example_dir("mobilenet")
    os.chdir(example_dir)
    sys.path.insert(0, example_dir)

    yield
    sys.path.remove(example_dir)


def test_mobilenet_qnn_ep():
    from mobilenet_qnn_ep import main as mobilenet_qnn_ep_main

    # need to pass [] since the parser reads from sys.argv
    result = mobilenet_qnn_ep_main([])

    expected_accelerator_spec = AcceleratorSpec(accelerator_type="npu", execution_provider="QNNExecutionProvider")
    # make sure it only ran for npu-qnn
    assert len(result) == 1
    assert expected_accelerator_spec in result
