# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from olive.cli.capture_onnx import CaptureOnnxGraphCommand


def _parse_capture_args(*args):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    CaptureOnnxGraphCommand.register_subcommand(commands)
    return parser.parse_args(["capture-onnx-graph", *args])


def test_capture_onnx_boolean_flags_default_to_disabled():
    args = _parse_capture_args()

    assert args.exclude_embeds is False
    assert args.exclude_lm_head is False
    assert args.enable_cuda_graph is None


def test_capture_onnx_boolean_flags_enable_on_presence():
    args = _parse_capture_args(
        "--exclude_embeds",
        "--exclude_lm_head",
        "--enable_cuda_graph",
    )

    assert args.exclude_embeds is True
    assert args.exclude_lm_head is True
    assert args.enable_cuda_graph is True
