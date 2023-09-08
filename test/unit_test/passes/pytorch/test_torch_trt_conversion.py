# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock, patch

import torch

import olive.passes.pytorch.sparsegpt_utils as sparsegpt_utils
from olive.common.utils import get_attr
from olive.data.template import huggingface_data_config_template
from olive.model import PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import TorchTRTConversion


class MockTRTLinearLayer(torch.nn.Module):
    pass


def mocked_torch_zeros(*args, **kwargs):
    if "device" in kwargs:
        del kwargs["device"]
    return torch.ones(*args, **kwargs) * 0


# all the following patches are needed to run the test on CPU
# make torch.cuda.is_available return True
@patch("torch.cuda.is_available", return_value=True)
# make tensor_data_to_device return the input tensor, ignore device
@patch("olive.common.utils.tensor_data_to_device", side_effect=lambda x, y: x)
# make torch.nn.Module.to do nothing
@patch("torch.nn.Module.to", side_effect=lambda device: None)
# replace device in kwargs with "cpu"
@patch("torch.zeros", side_effect=mocked_torch_zeros)
def test_torch_trt_conversion_success(
    mock_torch_zeros, mock_torch_nn_module_to, mock_tensor_data_to_device, mock_torch_cuda_is_available, tmp_path
):
    # setup
    # mock trt utils since we don't have tensorrt and torch-tensorrt installed
    mock_trt_utils = MagicMock()
    mock_compile_trt_model = MagicMock()
    mock_compile_trt_model.return_value = MockTRTLinearLayer()
    mock_trt_utils.compile_trt_model = mock_compile_trt_model
    # we don't want to import trt_utils because of missing tensorrt and torch-tensorrt
    # add mocked trt_utils to sys.modules
    sys.modules["olive.passes.pytorch.trt_utils"] = mock_trt_utils
    model_name = "hf-internal-testing/tiny-random-OPTForCausalLM"
    task = "text-generation"
    model_type = "opt"
    input_model = PyTorchModel(hf_config={"model_name": model_name, "task": task})
    # torch.nn.Linear submodules per layer in the original model
    original_submodules = list(
        sparsegpt_utils.get_layer_submodules(
            sparsegpt_utils.get_layers(input_model.load_model(), model_type)[0], submodule_types=[torch.nn.Linear]
        ).keys()
    )

    dataset = {
        "data_name": "ptb_text_only",
        "subset": "penn_treebank",
        "split": "train",
        "component_kwargs": {
            "pre_process_data": {
                "dataset_type": "corpus",
                "text_cols": ["sentence"],
                "corpus_strategy": "join-random",
                "source_max_len": 100,
                "max_samples": 1,
                "random_seed": 42,
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
    model = p.run(input_model, None, output_folder)

    # assert
    pytorch_model = model.load_model()
    layers = sparsegpt_utils.get_layers(pytorch_model, model_type)
    for layer in layers:
        for submodule_name in original_submodules:
            # check that the submodule is replaced with MockTRTLinearLayer
            assert isinstance(get_attr(layer, submodule_name), MockTRTLinearLayer)

    # cleanup
    del sys.modules["olive.passes.pytorch.trt_utils"]
