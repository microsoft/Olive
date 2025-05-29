# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch

from olive.hardware import AcceleratorSpec
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.capture_split_info import CaptureSplitInfo


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.before_layer = torch.nn.Linear(2, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])
        self.after_layers = torch.nn.ModuleList([torch.nn.Linear(4, 2) for _ in range(2)])

    def forward(self, x):
        x = self.before_layer(x)
        for layer in self.layers:
            x = layer(x)
        for layer in self.after_layers:
            x = layer(x)
        return x


@pytest.mark.parametrize(
    ("input_model", "block_to_split", "num_splits", "split_assignments", "unique_embeds_lm_head_splits"),
    [
        (
            PyTorchModelHandler(model_loader=lambda _: CustomModel()),
            "layers",
            2,
            {"layers.0": 0, "layers.1": 0, "layers.2": 1, "layers.3": 1},
            False,
        ),
        # Test not equally divide the axis
        (
            PyTorchModelHandler(model_loader=lambda _: CustomModel()),
            "layers",
            3,
            {"layers.0": 0, "layers.1": 0, "layers.2": 1, "layers.3": 2},
            False,
        ),
        (
            PyTorchModelHandler(model_loader=lambda _: CustomModel()),
            "",
            3,
            {"before_layer": 0, "layers": 1, "after_layers": 2},
            False,
        ),
        (
            PyTorchModelHandler(model_loader=lambda _: CustomModel()),
            ["layers", "after_layers"],
            2,
            {"layers.0": 0, "layers.1": 0, "layers.2": 0, "layers.3": 1, "after_layers.0": 1, "after_layers.1": 1},
            False,
        ),
        (
            HfModelHandler(model_path="hf-internal-testing/tiny-random-LlamaForCausalLM"),
            None,
            2,
            {"model.layers.0": 0, "model.layers.1": 1},
            False,
        ),
        (
            HfModelHandler(model_path="hf-internal-testing/tiny-random-LlamaForCausalLM"),
            None,
            2,
            {"model.embed_tokens": 0, "model.layers.0": 1, "model.layers.1": 2, "lm_head": 3},
            True,
        ),
    ],
)
def test_capture_split_info_num_splits(
    input_model, block_to_split, num_splits, split_assignments, unique_embeds_lm_head_splits, tmp_path
):
    config = {"num_splits": num_splits, "unique_embeds_lm_head_splits": unique_embeds_lm_head_splits}
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


@pytest.mark.parametrize("unique_embeds_lm_head", [True, False])
@pytest.mark.parametrize(("memory", "expected_num_splits"), [(1e4, 4), (2e6, 2)])
def test_capture_split_info_cost_model(memory, expected_num_splits, unique_embeds_lm_head, tmp_path):
    model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    cost_model_path = get_cost_model(tmp_path, model_name)

    p = create_pass_from_dict(
        CaptureSplitInfo,
        {"cost_model": cost_model_path, "unique_embeds_lm_head_splits": unique_embeds_lm_head},
        # 2 layers
        # each layer is around 10 kB in fp16
        # embed_tokens and lm_head are around 1 MB each in fp16
        accelerator_spec=AcceleratorSpec(accelerator_type="cpu", memory=memory),
        disable_search=True,
    )
    input_model = HfModelHandler(model_path=model_name)

    out = p.run(input_model, str(tmp_path))
    split_assignments = out.model_attributes["split_assignments"]

    if unique_embeds_lm_head:
        # 2 splits for embeddings and lm_head for 1e4 memory
        # 1 split for embeddings for 2e6 memory
        num_other_splits = max(1, expected_num_splits - 2)
        assert split_assignments["model.embed_tokens"] == 0
        assert split_assignments["lm_head"] == num_other_splits + 1
        expected_num_splits = num_other_splits + 2
    else:
        first_split_members = [k for k, v in split_assignments.items() if v == 0]
        assert first_split_members == ["model.embed_tokens"]

    assert len(set(split_assignments.values())) == expected_num_splits


def test_capture_split_info_missing_memory(tmp_path):
    p = create_pass_from_dict(
        CaptureSplitInfo,
        {
            "cost_model": "cost_model.csv",
        },
        disable_search=True,
    )
    input_model = HfModelHandler(model_path="hf-internal-testing/tiny-random-LlamaForCausalLM")

    with pytest.raises(ValueError, match="Accelerator memory is required to split using cost model."):
        p.run(input_model, tmp_path)
