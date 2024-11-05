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
    ("input_model", "block_to_split", "num_splits", "split_assignments"),
    [
        (
            PyTorchModelHandler(model_loader=lambda _: CustomModel()),
            "layers",
            2,
            {"layers.0": 0, "layers.1": 0, "layers.2": 1, "layers.3": 1},
        ),
        (
            PyTorchModelHandler(model_loader=lambda _: CustomModel()),
            "",
            3,
            {"before_layer": 0, "layers": 1, "after_layer": 2},
        ),
        (
            HfModelHandler(model_path="hf-internal-testing/tiny-random-LlamaForCausalLM"),
            None,
            2,
            {"model.layers.0": 0, "model.layers.1": 1},
        ),
    ],
)
def test_capture_split_info_num_splits(input_model, block_to_split, num_splits, split_assignments, tmp_path):
    config = {
        "num_splits": num_splits,
    }
    if block_to_split is not None:
        config["block_to_split"] = block_to_split
    p = create_pass_from_dict(CaptureSplitInfo, config, disable_search=True)

    out = p.run(input_model, str(tmp_path))
    assert out.model_attributes["split_assignments"] == split_assignments


def get_cost_model(tmp_path, model_name) -> str:
    from olive.cli.launcher import main as cli_main

    cost_model_path = str(tmp_path / "cost_model.csv")

    cli_main(["generate-cost-model", "-m", model_name, "-o", cost_model_path])

    return cost_model_path


@pytest.mark.parametrize("include_embeds_lm_head", [True, False])
def test_capture_split_info_cost_model(include_embeds_lm_head, tmp_path):
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    cost_model_path = get_cost_model(tmp_path, model_name)

    p = create_pass_from_dict(
        CaptureSplitInfo,
        {
            "cost_model": cost_model_path,
            # 2 layers
            # each layer is around 10 kB in fp16
            # embed_tokens and lm_head are around 1 MB each in fp16
            "max_memory": 1e4,
            "exclude_embeds": not include_embeds_lm_head,
            "exclude_lm_head": not include_embeds_lm_head,
        },
        disable_search=True,
    )
    input_model = HfModelHandler(model_path=model_name)

    out = p.run(input_model, str(tmp_path))
    split_assignments = out.model_attributes["split_assignments"]

    if not include_embeds_lm_head:
        assert "model.embed_tokens" not in split_assignments
        assert "model.lm_head" not in split_assignments
        expected_num_splits = 2
    else:
        expected_num_splits = 4

    assert len(set(split_assignments.values())) == expected_num_splits
