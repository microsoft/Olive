# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.hardware import AcceleratorSpec
from olive.model import CompositeModelHandler, HfModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.split import SplitModel
from olive.passes.onnx.transformer_optimization import OrtTransformersOptimization
from olive.passes.pytorch.capture_split_info import CaptureSplitInfo


# TODO(jambayk): Add model builder and qdq models to this test
@pytest.fixture(params=[True, False], ids=["cost_model", "num_splits"], name="input_model_info", scope="module")
def input_model_info_fixture(request, tmp_path_factory):
    # this tmp_path exists for the duration of the test session
    # module is scope is used to ensure that the fixture is only created once
    tmp_path = tmp_path_factory.mktemp("test-split-model")

    # store onnx models for use in tests
    all_models = {}

    # input model
    model_name = "katuni4ka/tiny-random-phi3"
    input_model = HfModelHandler(
        model_path="katuni4ka/tiny-random-phi3",
        load_kwargs={"trust_remote_code": False, "revision": "585361abfee667f3c63f8b2dc4ad58405c4e34e2"},
    )

    # add split info to the model
    capture_config = {}
    accelerator_spec = None
    if request.param:
        from olive.cli.launcher import main as cli_main

        cost_model_path = str(tmp_path / "cost_model.csv")

        cli_main(["generate-cost-model", "-m", model_name, "-o", cost_model_path])

        capture_config["cost_model"] = cost_model_path
        accelerator_spec = AcceleratorSpec(accelerator_type="cpu", memory=4e4)
    else:
        capture_config["num_splits"] = 2
    input_model = create_pass_from_dict(
        CaptureSplitInfo, capture_config, accelerator_spec=accelerator_spec, disable_search=True
    ).run(input_model, tmp_path)

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

    return all_models, request.param, 4 if request.param else 2


@pytest.mark.parametrize(
    "model_type",
    ["convert_fp32", "opt_fp32", "opt_fp16"],
)
def test_split_model_all_nodes(tmp_path, input_model_info, model_type):
    input_model_info, use_cost_model, expected_splits = input_model_info

    input_model = input_model_info[model_type]

    split_model = create_pass_from_dict(SplitModel, disable_search=True).run(input_model, tmp_path)

    assert isinstance(split_model, CompositeModelHandler)
    components = list(split_model.model_components)
    assert len(components) == expected_splits
    assert all(isinstance(component, ONNXModelHandler) for component in components)

    # check that the split models can be loaded
    for component in components:
        component.prepare_session(execution_providers="CPUExecutionProvider")

    if not use_cost_model:
        # check that the splits have the expected kv inputs
        for split_idx in range(2):
            io_config = components[split_idx].io_config

            expected_kv_inputs = [f"past_key_values.{split_idx}.key", f"past_key_values.{split_idx}.value"]
            assert set(expected_kv_inputs) <= set(io_config["input_names"])

            expected_kv_outputs = [f"present.{split_idx}.key", f"present.{split_idx}.value"]
            assert set(expected_kv_outputs) <= set(io_config["output_names"])

    # check that the splits are connected correctly and produce the expected outputs
    input_io_config = input_model.io_config
    model_inputs = set(input_io_config["input_names"])
    model_outputs = set(input_io_config["output_names"])

    seen_outputs = set()
    used_outputs = set()
    for idx in range(expected_splits):
        # input to split must be a subset of the model input and previous outputs
        assert set(components[idx].io_config["input_names"]) <= model_inputs | seen_outputs
        seen_outputs |= set(components[idx].io_config["output_names"])
        used_outputs |= set(components[idx].io_config["input_names"]) - model_inputs

        if idx == expected_splits - 1:
            # output of last split must be a subset of the output of the input model
            assert set(components[idx].io_config["output_names"]) <= model_outputs
    # all non model outputs must be used between the splits
    assert (seen_outputs - used_outputs) == model_outputs
