# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from pathlib import Path

import pytest
from utils import check_output


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    CIFAR10_DIR = str(Path(__file__).resolve().parent.parent / "cifar10_openvino_intel_hw")
    sys.path.append(CIFAR10_DIR)
    yield
    sys.path.remove(CIFAR10_DIR)


def test_cifar10():
    import cifar10

    metrics = cifar10.main()
    check_output(metrics)
