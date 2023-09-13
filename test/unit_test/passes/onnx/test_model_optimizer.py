# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_onnx_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OnnxModelOptimizer


def test_onnx_model_optimizer_pass(tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxModelOptimizer, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)
