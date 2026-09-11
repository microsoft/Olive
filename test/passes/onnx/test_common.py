# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os

import numpy as np
import onnx
import pytest

from olive.common.utils import is_hardlink
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import (
    add_version_metadata_to_model_proto,
    get_external_data_file_names,
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
        assert not is_hardlink(resave_path.parent / "resave.onnx.data")

    input_model = onnx.load(input_model.model_path)
    resaved_model = onnx.load(resave_path)

    if not is_hardlink(resave_path):
        input_model = add_version_metadata_to_model_proto(input_model)

    assert resaved_model == input_model


def test_resave_model_preserves_multiple_external_files(tmp_path):
    input_path = tmp_path / "input" / "model.onnx"
    input_path.parent.mkdir()
    first = onnx.numpy_helper.from_array(np.ones((2, 2), dtype="float32"), "first")
    second = onnx.numpy_helper.from_array(np.ones((2, 2), dtype="float32"), "second")
    graph = onnx.helper.make_graph([], "external", [], [], initializer=[first, second])
    model = onnx.helper.make_model(graph)
    onnx.save_model(
        model,
        input_path,
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=0,
    )
    input_external_files = get_external_data_file_names(input_path)
    os.link(input_path.parent / input_external_files[0], tmp_path / "hardlink")

    output_path = tmp_path / "output" / "model.onnx"
    resave_model(input_path, output_path)

    external_files = get_external_data_file_names(output_path)
    assert len(external_files) == 2
    assert all((output_path.parent / name).exists() for name in external_files)
    assert all(not is_hardlink(output_path.parent / name) for name in external_files)
    onnx.load(output_path)
