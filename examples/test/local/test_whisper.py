# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import platform
import sys

import pytest

from olive.common.constants import OS

from ..utils import check_output, get_example_dir


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    example_dir = get_example_dir("whisper")
    os.chdir(example_dir)
    sys.path.insert(0, example_dir)

    # prepare configs
    from prepare_whisper_configs import main as prepare_whisper_configs

    prepare_whisper_configs(["--package_model", "--log_level", "0"])

    yield
    sys.path.remove(example_dir)


@pytest.mark.parametrize("device_precision", [("cpu", "fp32"), ("cpu", "int8"), ("cpu", "inc_int8")])
def test_whisper(device_precision):
    from olive.workflows import run as olive_run

    if platform.system() == OS.WINDOWS and device_precision[1].startswith("inc_int8"):
        pytest.skip("Skip test on Windows. neural-compressor import is hanging on Windows.")

    device, precision = device_precision
    config_file = f"whisper_{device}_{precision}.json"
    with open(config_file) as f:
        olive_config = json.load(f)

    # test workflow
    result = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(result)

    # test transcription
    from test_transcription import main as test_transcription

    transcription = test_transcription(["--config", config_file])
    assert len(transcription) > 0
