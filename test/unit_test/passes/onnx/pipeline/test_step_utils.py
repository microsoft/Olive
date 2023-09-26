# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from test.unit_test.passes.onnx.test_pre_post_processing_op import (
    convert_superresolution_model,
    get_superresolution_model,
)

from olive.passes.onnx.pipeline.step_utils import parse_steps


class CustomizedParam:
    def __init__(self, params: dict):
        self.params = params


def test_step_parser(tmp_path):
    from onnxruntime_extensions.tools.pre_post_processing import TokenizerParam

    pytorch_model = get_superresolution_model()
    input_model = convert_superresolution_model(pytorch_model, tmp_path)
    model = input_model.load_model()

    step_config = Path(__file__).parent / "step_config.json"
    with step_config.open() as f:
        config = json.load(f)

    steps = parse_steps(model, config)
    steps = dict(steps)
    assert len(steps) == 8
    assert isinstance(steps["SentencePieceTokenizer"]["tokenizer_param"], TokenizerParam)
    # assert Resize tuple
    assert isinstance(steps["Resize"], tuple)
    # assert the resize_to is a tuple of int
    resize_to = steps["Resize"][0]["resize_to"]
    assert isinstance(resize_to, tuple)
    assert all(isinstance(item, int) for item in resize_to)
    # assert io_map is list of list
    assert isinstance(steps["Resize"][1], list)
    assert all(isinstance(item, list) for item in steps["Resize"][1])
    dummy_step = steps["Dummy1"]
    assert isinstance(dummy_step, tuple)
    assert len(dummy_step) == 2
    dummy_step_params = dummy_step[0]
    assert isinstance(dummy_step_params["explicit_tuple"], tuple)
    assert isinstance(dummy_step_params["explicit_placholder"], tuple)
    assert len(dummy_step_params["explicit_placholder"]) == 4
    assert isinstance(dummy_step_params["implicit_tuple"], tuple)
    assert isinstance(dummy_step_params["explicit_list"], list)
    dummy2_param = steps["Dummy2"]["dummy2_param"]
    assert isinstance(dummy2_param, CustomizedParam)
    assert dummy2_param.params["a"] == 1
