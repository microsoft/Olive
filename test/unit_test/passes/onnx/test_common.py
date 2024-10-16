# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from test.unit_test.utils import ONNX_MODEL_PATH

import onnx
import pytest

from olive.passes.onnx.common import model_proto_to_olive_model


@pytest.mark.parametrize(
    "external_data_config",
    [
        {},
        {"save_as_external_data": True},
        {
            "save_as_external_data": False,
            "all_tensors_to_one_file": True,
            "external_data_name": None,
            "size_threshold": 1024,
            "convert_attribute": False,
        },
    ],
)
def test_model_proto_to_olive_model(external_data_config, tmp_path):
    model_proto = onnx.load(ONNX_MODEL_PATH)
    olive_model = model_proto_to_olive_model(model_proto, tmp_path / "test.onnx", external_data_config)
    assert olive_model, "Failed to save ONNX proto to Olive model"
