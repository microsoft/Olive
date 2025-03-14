# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from test.unit_test.utils import make_local_tiny_llama

import pytest

from olive.data.component.dataloader import LLMAugmentedDataLoader
from olive.data.template import huggingface_data_config_template
from olive.passes.olive_pass import create_pass_from_dict


@pytest.mark.parametrize("use_gqa", [True, False])
def test_llm_augmented_dataloader(tmp_path, use_gqa):
    pytorch_model = make_local_tiny_llama(tmp_path)
    if use_gqa:
        from olive.passes.onnx.model_builder import ModelBuilder

        onnx_model = create_pass_from_dict(ModelBuilder, {"precision": "fp32"}, disable_search=True).run(
            pytorch_model, tmp_path / "onnx_model"
        )
    else:
        from olive.passes.onnx.conversion import OnnxConversion

        onnx_model = create_pass_from_dict(OnnxConversion, {}, disable_search=True).run(
            pytorch_model, tmp_path / "onnx_model"
        )

    data_config = huggingface_data_config_template(
        model_name=pytorch_model.model_name_or_path,
        task="text-generation",
        load_dataset_config={"data_name": "wikitext", "subset": "wikitext-2-raw-v1", "split": "train"},
        pre_process_data_config={"add_special_tokens": False, "max_seq_len": 10, "max_samples": 1},
    )

    augmented_dataloader = LLMAugmentedDataLoader(
        data_config.to_data_container().create_dataloader(),
        model_path=onnx_model.model_path,
        io_config=onnx_model.io_config,
    )

    assert len(augmented_dataloader) == 1 if use_gqa else 2

    for idx, (batch, _) in enumerate(augmented_dataloader):
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "past_key_values.0.key" in batch
        assert "past_key_values.0.value" in batch
        if not use_gqa:
            assert "position_ids" in batch
            assert batch["past_key_values.0.key"].shape[2] == (1 if idx % 2 == 0 else 10)
        else:
            assert batch["past_key_values.0.key"].shape[2] == 0
