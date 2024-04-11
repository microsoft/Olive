# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys
from pathlib import Path

import pytest

from olive.hardware import AcceleratorSpec


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = str(cur_dir / "mobilenet")
    os.chdir(example_dir)
    sys.path.insert(0, example_dir)

    yield
    os.chdir(cur_dir)
    sys.path.remove(example_dir)


def test_mobilenet_qnn_ep():
    from mobilenet_qnn_ep import main as mobilenet_qnn_ep_main

    result = mobilenet_qnn_ep_main(["--skip_data_download"])

    expected_accelerator_spec = AcceleratorSpec(accelerator_type="npu", execution_provider="QNNExecutionProvider")
    # make sure it only ran for npu-qnn
    assert len(result) == 1
    assert expected_accelerator_spec in result
