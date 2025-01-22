# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

import pytest
import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.utils import get_attr
from olive.data.template import huggingface_data_config_template
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.sparsegpt_utils import get_layer_submodules
from olive.passes.pytorch.torch_trt_conversion import TorchTRTConversion

# pylint: disable=abstract-method


class MockTRTLinearLayer(torch.nn.Module):
    pass


@pytest.fixture(name="mock_torch_ort")
def mock_torch_ort_fixture():
    # mock trt utils since we don't have tensorrt and torch-tensorrt installed
    mock_trt_utils = MagicMock()
    mock_compile_trt_model = MagicMock()
    mock_compile_trt_model.return_value = MockTRTLinearLayer()
    mock_trt_utils.compile_trt_model = mock_compile_trt_model
    # we don't want to import trt_utils because of missing tensorrt and torch-tensorrt
    # add mocked trt_utils to sys.modules
    sys.modules["olive.passes.pytorch.trt_utils"] = mock_trt_utils
    yield mock_trt_utils
    del sys.modules["olive.passes.pytorch.trt_utils"]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TorchTRTConversion requires GPU.",
)
@pytest.mark.usefixtures("mock_torch_ort")
def test_torch_trt_conversion_success(tmp_path):
    # setup
    model_name = "hf-internal-testing/tiny-random-OPTForCausalLM"
    task = "text-generation"
    input_model = HfModelHandler(model_path=model_name, task=task)
    # torch.nn.Linear submodules per layer in the original model
    original_submodules = list(
        get_layer_submodules(
            ModelWrapper.from_model(input_model.load_model()).get_layers(False)[0], submodule_types=[torch.nn.Linear]
        ).keys()
    )

    dataset = {
        "load_dataset_config": {
            "params": {
                "data_name": "ptb_text_only",
                "subset": "penn_treebank",
                "split": "train",
                "trust_remote_code": True,
            }
        },
        "pre_process_data_config": {
            "params": {
                "text_cols": ["sentence"],
                "strategy": "join-random",
                "max_seq_len": 100,
                "max_samples": 1,
                "random_seed": 42,
                "trust_remote_code": True,
            }
        },
    }
    data_config = huggingface_data_config_template(model_name=model_name, task=task, **dataset)
    config = {
        "data_config": data_config,
    }

    p = create_pass_from_dict(TorchTRTConversion, config, disable_search=True)
    output_folder = str(tmp_path / "sparse")

    # execute
    model = p.run(input_model, output_folder)

    # assert
    for layer in ModelWrapper.from_model(model.load_model()).get_layers(False):
        for submodule_name in original_submodules:
            # check that the submodule is replaced with MockTRTLinearLayer
            assert isinstance(get_attr(layer, submodule_name), MockTRTLinearLayer)
