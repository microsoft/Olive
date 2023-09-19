# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import platform
import sys
from pathlib import Path

import pytest
from utils import check_output


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = str(cur_dir / "whisper")
    os.chdir(example_dir)
    sys.path.append(example_dir)

    # prepare configs
    from prepare_whisper_configs import main as prepare_whisper_configs

    prepare_whisper_configs([])

    yield
    os.chdir(cur_dir)
    sys.path.remove(example_dir)


@pytest.mark.parametrize("device_precision", [("cpu", "fp32"), ("cpu", "int8"), ("cpu", "inc_int8")])
def test_whisper(device_precision):
    from olive.workflows import run as olive_run

    if platform.system() == "Windows" and device_precision[1].startswith("inc_int8"):
        pytest.skip("Skip test on Windows. neural-compressor import is hanging on Windows.")

    device, precision = device_precision
    config_file = f"whisper_{device}_{precision}.json"
    olive_config = json.load(open(config_file, "r"))

    # test workflow
    result = olive_run(olive_config)
    check_output(result)

    # test transcription
    from test_transcription import main as test_transcription

    transcription = test_transcription(["--config", config_file])
    print(transcription)
    assert len(transcription) > 0
