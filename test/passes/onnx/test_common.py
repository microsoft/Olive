# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import onnx
import onnx_ir as ir
import pytest

from olive.common.utils import is_hardlink
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import (
    add_version_metadata_to_model_proto,
    ir_model_to_olive_model,
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
        {"save_as_external_data": True, "all_tensors_to_one_file": False, "size_threshold": 0},
        {"save_as_external_data": True, "convert_attribute": True, "size_threshold": 0},
    ],
)
@pytest.mark.parametrize("use_ir", [False, True])
def test_model_to_olive_model(external_data_config, use_ir, tmp_path):
    model_proto = onnx.load(ONNX_MODEL_PATH)
    if external_data_config.get("convert_attribute"):
        model_proto.graph.node.append(
            onnx.helper.make_node(
                "Constant",
                [],
                ["constant_output"],
                value=onnx.numpy_helper.from_array(np.ones(2, dtype=np.float32)),
            )
        )

    if use_ir:
        olive_model = ir_model_to_olive_model(ir.from_proto(model_proto), tmp_path / "test.onnx", external_data_config)
    else:
        olive_model = model_proto_to_olive_model(model_proto, tmp_path / "test.onnx", external_data_config)

    saved_model = onnx.load(olive_model.model_path, load_external_data=False)
    initializer_locations = {
        entry.value
        for initializer in saved_model.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    }
    attribute_locations = {
        entry.value
        for node in saved_model.graph.node
        for attribute in node.attribute
        if attribute.type == onnx.AttributeProto.TENSOR
        for entry in attribute.t.external_data
        if entry.key == "location"
    }

    if external_data_config.get("size_threshold") == 0:
        if external_data_config.get("all_tensors_to_one_file") is False:
            assert initializer_locations == {initializer.name for initializer in saved_model.graph.initializer}
        else:
            assert initializer_locations == {"test.onnx.data"}
    else:
        assert not initializer_locations

    if external_data_config.get("convert_attribute"):
        assert attribute_locations == {"test.onnx.data"}
    else:
        assert not attribute_locations


@pytest.mark.parametrize("has_external_data", [True, False])
def test_resave_model(has_external_data, tmp_path):
    # setup
    from transformers.cache_utils import DynamicLayer

    original_lazy_initialization = DynamicLayer.lazy_initialization
    input_model = create_pass_from_dict(
        OnnxConversion, {"save_as_external_data": has_external_data, "use_dynamo_exporter": True}, disable_search=True
    ).run(get_hf_model(), str(tmp_path / "input"))
    assert DynamicLayer.lazy_initialization is original_lazy_initialization

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
