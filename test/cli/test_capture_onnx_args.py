# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

import pytest

from olive.cli.capture_onnx import CaptureOnnxGraphCommand


def _parse_capture_args(*args):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    CaptureOnnxGraphCommand.register_subcommand(commands)
    return parser.parse_args(["capture-onnx-graph", *args])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_capture_onnx_boolean_arguments(value, expected):
    args = _parse_capture_args(
        "--exclude_embeds",
        value,
        "--exclude_lm_head",
        value,
        "--enable_cuda_graph",
        value,
    )

    assert args.exclude_embeds is expected
    assert args.exclude_lm_head is expected
    assert args.enable_cuda_graph is expected


def test_capture_onnx_boolean_arguments_reject_invalid_value():
    with pytest.raises(SystemExit):
        _parse_capture_args("--exclude_embeds", "not-a-bool")
