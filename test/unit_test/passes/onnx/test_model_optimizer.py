# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_onnx_model

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OnnxModelOptimizer


def test_onnx_model_optimizer_pass(tmpdir):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxModelOptimizer, {}, disable_search=True)
    output_folder = str(Path(tmpdir) / "onnx")

    # execute
    p.run(input_model, None, output_folder)
