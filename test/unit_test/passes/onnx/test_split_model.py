# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.model import CompositeModelHandler, HfModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.split import SplitModel
from olive.passes.onnx.transformer_optimization import OrtTransformersOptimization


# TODO(jambayk): Add model builder and qdq models to this test
@pytest.fixture(name="input_model_info", scope="module")
def input_model_info_fixture(tmp_path_factory):
    # this tmp_path exists for the duration of the test session
    # module is scope is used to ensure that the fixture is only created once
    tmp_path = tmp_path_factory.mktemp("test-split-model")

    # store onnx models for use in tests
    all_models = {}

    # input model
    input_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        load_kwargs={"trust_remote_code": False, "revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"},
        model_attributes={"split_assignments": {"model.layers.0": 0, "model.layers.1": 1}},
    )

    # conversion fp32
    all_models["convert_fp32"] = create_pass_from_dict(
        OnnxConversion, {"torch_dtype": "float32"}, disable_search=True
    ).run(input_model, tmp_path / "convert_fp32")
    # transformers opt fp32
    all_models["opt_fp32"] = create_pass_from_dict(
        OrtTransformersOptimization, {"model_type": "bert", "opt_level": 0}, disable_search=True
    ).run(all_models["convert_fp32"], tmp_path / "opt_fp32")
    # transformers opt fp16
    all_models["opt_fp16"] = create_pass_from_dict(
        OrtTransformersOptimization,
        {"model_type": "bert", "opt_level": 0, "float16": True, "keep_io_types": False},
        disable_search=True,
    ).run(all_models["convert_fp32"], tmp_path / "opt_fp16")
    # transformers opt fp16 with keep_io_types
    all_models["opt_fp16_keep_io_types"] = create_pass_from_dict(
        OrtTransformersOptimization,
        {"model_type": "bert", "opt_level": 0, "float16": True, "keep_io_types": True},
        disable_search=True,
    ).run(all_models["convert_fp32"], tmp_path / "opt_fp16_keep_io_types")

    return all_models


@pytest.mark.parametrize(
    ("model_type", "expected_overlap"),
    [
        ("convert_fp32", 3),
        ("opt_fp32", 4),
        ("opt_fp16", 4),
        ("opt_fp16_keep_io_types", 4),
    ],
)
def test_split_model(tmp_path, input_model_info, model_type, expected_overlap):
    input_model = input_model_info[model_type]

    split_model = create_pass_from_dict(SplitModel, disable_search=True).run(input_model, tmp_path)

    assert isinstance(split_model, CompositeModelHandler)
    components = list(split_model.model_components)
    assert len(components) == 2
    assert all(isinstance(component, ONNXModelHandler) for component in components)

    # check that the splits have the expected kv inputs
    for split_idx in range(2):
        io_config = components[split_idx].io_config

        expected_kv_inputs = [f"past_key_values.{split_idx}.key", f"past_key_values.{split_idx}.value"]
        if model_type == "opt_fp16_keep_io_types":
            # key is used before the firt split too
            expected_kv_inputs.remove(f"past_key_values.{split_idx}.key")
        assert set(expected_kv_inputs) <= set(io_config["input_names"])

        expected_kv_outputs = [f"present.{split_idx}.key", f"present.{split_idx}.value"]
        assert set(expected_kv_outputs) <= set(io_config["output_names"])
    # check that the splits have the expected overlap
    assert (
        len(set(components[0].io_config["output_names"]).intersection(set(components[1].io_config["input_names"])))
        == expected_overlap
    )
    # check that the split models can be loaded
    # TODO(jambayk): Consider running the full model and comparing the outputs with the splits
    # this is more involved. have to modify the input model to create outputs for the split outputs
    for component in components:
        component.prepare_session(execution_providers="CPUExecutionProvider")
