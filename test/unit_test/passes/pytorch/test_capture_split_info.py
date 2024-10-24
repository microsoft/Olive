# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch

from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.capture_split_info import CaptureSplitInfo


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.before_layer = torch.nn.Linear(2, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
        self.after_layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.before_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.after_layer(x)


@pytest.mark.parametrize(
    ("input_model", "layers_block_name", "split_assignments"),
    [
        (
            PyTorchModelHandler(model_loader=lambda _: CustomModel()),
            "layers",
            {"layers.0": 0, "layers.1": 0, "layers.2": 1, "layers.3": 1},
        ),
        (
            HfModelHandler(model_path="hf-internal-testing/tiny-random-LlamaForCausalLM"),
            None,
            {"model.layers.0": 0, "model.layers.1": 1},
        ),
    ],
)
def test_capture_split_info(input_model, layers_block_name, split_assignments, tmp_path):
    config = {
        "num_splits": 2,
    }
    if layers_block_name:
        config["layers_block_name"] = layers_block_name
    p = create_pass_from_dict(CaptureSplitInfo, config, disable_search=True)

    out = p.run(input_model, str(tmp_path))
    assert out.model_attributes["split_assignments"] == split_assignments
