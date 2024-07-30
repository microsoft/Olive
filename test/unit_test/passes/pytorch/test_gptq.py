# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from olive.data.config import DataComponentConfig, DataConfig
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.gptq import GptqQuantizer


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="gptq requires GPU.",
)
def test_gptq_default(tmp_path: Path):
    # setup
    input_model = HfModelHandler(model_path="facebook/opt-125m")
    config = {
        "data_config": DataConfig(
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
        ),
    }
    p = create_pass_from_dict(
        GptqQuantizer,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    out = p.run(input_model, gptq_out_folder)
    assert out is not None
