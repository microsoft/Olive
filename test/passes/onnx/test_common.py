# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnx
import pytest

from olive.common.utils import is_hardlink
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import (
    add_version_metadata_to_model_proto,
    model_proto_to_olive_model,
    resave_model,
)
from olive.passes.onnx.conversion import OnnxConversion
from test.utils import ONNX_MODEL_PATH, get_hf_model


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


@pytest.mark.parametrize("has_external_data", [True, False])
def test_resave_model(has_external_data, tmp_path):
    # setup
    input_model = create_pass_from_dict(
        OnnxConversion, {"save_as_external_data": has_external_data, "use_dynamo_exporter": True}, disable_search=True
    ).run(get_hf_model(), str(tmp_path / "input"))

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
