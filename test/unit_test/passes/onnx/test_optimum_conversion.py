# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from test.unit_test.utils import get_hf_model

import pytest

from olive.model import CompositeModelHandler, HfModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.optimum_conversion import OptimumConversion


@pytest.mark.parametrize("extra_args", [{"atol": 0.1}, {"atol": None}])
def test_optimum_conversion_pass(extra_args, tmp_path):
    input_model = get_hf_model()
    # setup
    p = create_pass_from_dict(OptimumConversion, {"extra_args": extra_args}, disable_search=True)
    output_folder = tmp_path

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    assert Path(onnx_model.model_path).exists()


@pytest.mark.parametrize(
    ("components", "extra_args", "expected_components"),
    [
        (None, None, None),  # latest model can be used for both prompt processing and token generation
        (
            None,
            {"legacy": True, "no_post_process": True},
            ["decoder_with_past_model", "decoder_model"],
        ),  # legacy model with separate prompt processing and token generation
        (
            ["decoder_model", "decoder_with_past_model"],
            {"legacy": True, "no_post_process": True},
            ["decoder_model", "decoder_with_past_model"],
        ),  # select specific components in order from legacy export
    ],
)
def test_optimum_conversion_pass_with_components(components, extra_args, expected_components, tmp_path):
    input_model = HfModelHandler(model_path="hf-internal-testing/tiny-random-OPTForCausalLM")
    # setup
    p = create_pass_from_dict(
        OptimumConversion, {"components": components, "extra_args": extra_args}, disable_search=True
    )
    output_folder = tmp_path

    # execute
    output_model = p.run(input_model, output_folder)

    # assert
    if expected_components is None:
        # for latest optimum versions like 1.16.1
        if isinstance(output_model, ONNXModelHandler):
            assert Path(output_model.model_path).exists()
        else:
            # for older optimum versions like 1.13.2
            assert isinstance(output_model, CompositeModelHandler)
    else:
        assert isinstance(output_model, CompositeModelHandler)
        component_names = []
        for component_name, component_model in output_model.get_model_components():
            component_names.append(component_name)
            assert Path(component_model.model_path).exists()
        assert set(component_names) == set(expected_components)


@pytest.mark.parametrize(
    ("config", "is_valid"),
    [
        ({"fp16": True}, False),
        ({"fp16": True, "device": "cpu"}, False),
        ({"fp16": False, "device": "cpu"}, True),
    ],
)
def test_optimum_configs(config, is_valid, tmp_path):
    input_model = get_hf_model()
    # setup
    p = create_pass_from_dict(OptimumConversion, config, disable_search=True)
    output_folder = tmp_path

    if not is_valid:
        assert p.validate_config(p.config, None) is False
        with pytest.raises(
            ValueError,
            match="FP16 export is supported only when exporting on GPU. Please pass the option `--device cuda`.",
        ):
            p.run(input_model, output_folder)
    else:
        assert p.validate_config(p.config, None) is True
        onnx_model = p.run(input_model, output_folder)
        assert Path(onnx_model.model_path).exists()
