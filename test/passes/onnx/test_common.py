# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnx
import pytest

from olive.common.utils import is_hardlink
from olive.model import ONNXModelHandler
from olive.passes.onnx.common import (
    add_version_metadata_to_model_proto,
    model_proto_to_olive_model,
    resave_model,
)
from test.utils import ONNX_MODEL_PATH


@pytest.mark.parametrize(
    "external_data_config",
    [
        {},
        {"save_as_external_data": True},
    ],
)
def test_model_proto_to_olive_model(external_data_config, tmp_path):
    model_proto = onnx.load(ONNX_MODEL_PATH)
    olive_model = model_proto_to_olive_model(model_proto, tmp_path / "test.onnx", external_data_config)
    assert olive_model, "Failed to save ONNX proto to Olive model"


@pytest.mark.parametrize("has_external_data", [True, False])
def test_resave_model(has_external_data, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_path = input_dir / "input.onnx"
    model_proto = onnx.load(ONNX_MODEL_PATH)
    if has_external_data:
        onnx.save_model(
            model_proto,
            input_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="input.onnx.data",
            size_threshold=0,
            convert_attribute=True,
        )
    else:
        onnx.save(model_proto, input_path)
    input_model = ONNXModelHandler(input_path)

    # execute
    resave_path = tmp_path / "resave" / "resave.onnx"
    resave_model(input_model.model_path, resave_path)

    # assert
    assert resave_path.exists()
    if has_external_data:
        assert (resave_path.parent / "resave.onnx.data").exists()

    input_model = onnx.load(input_model.model_path)
    resaved_model = onnx.load(resave_path)

    if not is_hardlink(resave_path):
        input_model = add_version_metadata_to_model_proto(input_model)

    assert resaved_model == input_model
