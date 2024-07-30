# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_onnx_model

import onnx
import pytest

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.float16_conversion import OnnxFloatToFloat16


@pytest.mark.parametrize("keep_io_types", [True, False])
def test_onnxfloattofloat16(keep_io_types, tmp_path):
    # setup
    # this is a simple model with a single Gemm node
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxFloatToFloat16, {"keep_io_types": keep_io_types}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    output_model = p.run(input_model, output_folder)

    # assert
    # check that the input and output types are as expected
    io_config = output_model.io_config
    for io_type in [*io_config["input_types"], *io_config["output_types"]]:
        assert io_type == ("float32" if keep_io_types else "float16")

    # check that the model initializer types are float16
    for initializer in output_model.get_graph().initializer:
        assert initializer.data_type == onnx.TensorProto.FLOAT16
