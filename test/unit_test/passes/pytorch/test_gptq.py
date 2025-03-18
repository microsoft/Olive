# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import make_local_tiny_llama

import pytest
import torch

from olive.data.config import DataComponentConfig, DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.gptq import GptqQuantizer

test_gptq_dc_config = DataConfig(
    name="test_gptq_dc_config",
    type="DummyDataContainer",
    load_dataset_config=DataComponentConfig(
        type="dummy_dataset",
        params={
            "input_names": ["input_ids", "attention_mask"],
            "input_shapes": [[1, 128], [1, 128]],
            "input_types": ["int64", "int64"],
            "max_samples": 128,
        },
    ),
    pre_process_data_config=DataComponentConfig(type="skip_pre_process"),
    post_process_data_config=DataComponentConfig(type="skip_post_process"),
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="gptq requires GPU.",
)
@pytest.mark.parametrize(
    ("model_path", "expected_model_type", "data_config"),
    [
        ("katuni4ka/tiny-random-phi3", "Phi3ForCausalLM", None),
        ("katuni4ka/tiny-random-phi3", "Phi3ForCausalLM", test_gptq_dc_config),
        ("tiny-llama", "LlamaForCausalLM", None),
    ],
)
def test_gptq_default(tmp_path: Path, model_path: str, expected_model_type: str, data_config: DataConfig):
    # setup
    if model_path == "tiny-llama":
        input_model = make_local_tiny_llama(tmp_path / "input_model")
    else:
        input_model = HfModelHandler(model_path=model_path)
    p = create_pass_from_dict(
        GptqQuantizer,
        {"data_config": data_config},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    out = p.run(input_model, gptq_out_folder)

    # assert
    assert isinstance(out, HfModelHandler)
    assert out.load_model().__class__.__name__ == expected_model_type
