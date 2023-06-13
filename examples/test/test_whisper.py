# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from pathlib import Path

import pytest
from utils import check_no_eval_output, check_no_search_output

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


@pytest.mark.parametrize("custom_device_precision", [(True, "cpu", "fp32"), (True, "cpu", "int8"), (False, None, None)])
def test_whisper(custom_device_precision):
    from olive.workflows import run as olive_run

    is_custom, device, precision = custom_device_precision
    if is_custom:
        olive_config = json.load(open(f"whisper_{device}_{precision}.json", "r"))
    else:
        olive_config = json.load(open("whisper.json", "r"))

    result = olive_run(olive_config)
    if is_custom:
        check_no_search_output(result)
    else:
        check_no_eval_output(result)
