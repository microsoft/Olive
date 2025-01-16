# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch

from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.rotate import QuaRot


@pytest.mark.parametrize(
    "model_path", ["katuni4ka/tiny-random-phi3", "hf-internal-testing/tiny-random-LlamaForCausalLM"]
)
@pytest.mark.parametrize("rotate_mode", ["hadamard", "random"])
def test_quarot(tmp_path, model_path, rotate_mode):
    input_model = HfModelHandler(model_path=model_path)
    if input_model.get_hf_model_type() == "llama":
        # this checkpoint has an invalid generation config that cannot be saved
        model_path = str(tmp_path / "model")
        loaded_model = input_model.load_model()
        loaded_model.generation_config.pad_token_id = 1
        loaded_model.save_pretrained(model_path)
        input_model.get_hf_tokenizer().save_pretrained(model_path)
        input_model = HfModelHandler(model_path=model_path)

    p = create_pass_from_dict(QuaRot, {"rotate_mode": rotate_mode}, disable_search=True)

    output_path = str(tmp_path / "output")
    output_model = p.run(input_model, output_path)

    assert isinstance(output_model, HfModelHandler)
    assert input_model.model_path == model_path
    assert output_model.model_path == output_path

    original_model = input_model.load_model()
    rotated_model = output_model.load_model()

    i = torch.randint(0, 10, (1, 2))
    with torch.no_grad():
        original_output = original_model(i)
        rotated_output = rotated_model(i)
        assert torch.allclose(original_output.logits, rotated_output.logits, atol=1e-5)
