# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest

from olive.constants import Precision
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.vitis_ai.meta_data import VitisAIAddMetaData
from test.utils import get_onnx_model


@pytest.mark.parametrize(
    ("a_type", "w_type", "expected_a_type", "expected_w_type", "valid"),
    [
        (Precision.UINT16, Precision.UINT8, "QUInt16", "QUInt8", True),
        (Precision.UINT16, Precision.INT4, "QUInt16", "Int4", True),
        (Precision.UINT8, Precision.UINT8, "QUInt8", "QUInt8", True),
        (None, Precision.INT8, None, "QInt8", True),
        (Precision.INT16, None, "QInt16", None, True),
        (Precision.NF4, Precision.UINT8, "Invalid", "QUInt8", False),
    ],
)
def test_vitis_ai_add_metadata(a_type, w_type, expected_a_type, expected_w_type, valid, tmp_path):
    # setup
    input_model = get_onnx_model(
        model_attributes={
            "architectures": ["TestArchitecture"],
            "model_type": "test_model_type",
            "test_key": "test_value",
        }
    )
    p = create_pass_from_dict(
        VitisAIAddMetaData,
        {
            "config_meta_data_keys": ["architectures", "model_type"],
            "activation_type": a_type,
            "weight_type": w_type,
            "quant_type": "OnnxStaticQuantization",
        },
        disable_search=True,
    )
    assert p.validate_config(p.config, None) == valid
    if not valid:
        return

    output_model = p.run(input_model, str(tmp_path))

    metatadata_props = {entry.key: entry.value for entry in output_model.load_model().metadata_props}
    assert metatadata_props["architectures"] == "TestArchitecture"
    assert metatadata_props["model_type"] == "test_model_type"
    if a_type is None:
        assert "activation_dtype" not in metatadata_props
    else:
        assert metatadata_props["activation_dtype"] == expected_a_type
    if w_type is None:
        assert "weight_dtype" not in metatadata_props
    else:
        assert metatadata_props["weight_dtype"] == expected_w_type
    assert metatadata_props["quant_type"] == "OnnxStaticQuantization"
