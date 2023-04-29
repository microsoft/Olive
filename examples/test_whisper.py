# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from pathlib import Path

import pytest

from olive.common.utils import run_subprocess


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = str(Path(__file__).resolve().parent)
    example_dir = str(Path(__file__).resolve().parent / "whisper")
    os.chdir(example_dir)

    # prepare configs
    try:
        run_subprocess("python prepare_configs.py")
    except Exception:
        # for some reason the first time fails on windows
        run_subprocess("python prepare_configs.py")

    yield
    os.chdir(cur_dir)


def check_output(output):
    assert output["metrics"]["latency"] > 0


@pytest.mark.parametrize("device_precision", [("cpu", "fp32"), ("cpu", "int8")])
def test_whisper(device_precision):
    from olive.workflows import run as olive_run

    device, precision = device_precision
    olive_config = json.load(open(f"whisper_{device}_{precision}.json", "r"))

    result = olive_run(olive_config)
    check_output(result)
