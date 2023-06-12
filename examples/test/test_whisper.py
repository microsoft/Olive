# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from pathlib import Path

import pytest
from utils import check_no_search_output

from olive.common.utils import retry_func, run_subprocess


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "whisper"
    os.chdir(example_dir)

    # prepare configs
    # retry since it fails randomly on windows
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_whisper_configs.py", "check": True})

    yield
    os.chdir(cur_dir)


@pytest.mark.parametrize("device_precision", [("cpu", "fp32"), ("cpu", "int8")])
def test_whisper(device_precision):
    from olive.workflows import run as olive_run

    device, precision = device_precision
    olive_config = json.load(open(f"whisper_{device}_{precision}.json", "r"))

    result = olive_run(olive_config)
    check_no_search_output(result)
