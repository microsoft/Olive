# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib.util
import pathlib
import sys
import types
from unittest.mock import patch

import torch
import torch.nn as nn

from olive.common.quant.tensor import QuantTensor


def _load_nemotron_module():
    module_path = pathlib.Path(__file__).resolve().parents[2] / "playground" / "nemotron" / "model_loader.py"
    spec = importlib.util.spec_from_file_location("nemotron_model_loader", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nemotron_asr_model_loader_uses_local_state_dict_when_path_is_directory(tmp_path):
    module = _load_nemotron_module()
    model_dir = tmp_path / "quantized_model"
    model_dir.mkdir()
    state_dict_path = model_dir / "model.pt"
    expected_state = {"encoder.weight": torch.tensor([1.0, 2.0])}
    torch.save(expected_state, state_dict_path)

    class FakeASRModel:
        def __init__(self):
            self.loaded_state = None

        def cpu(self):
            return self

        def eval(self):
            return self

        @classmethod
        def from_pretrained(cls, model_name=None, **kwargs):
            if model_name == "nvidia/nemotron-speech-streaming-en-0.6b":
                return cls()
            raise AssertionError(f"Unexpected model_name: {model_name!r}")

        def load_state_dict(self, state_dict, strict=True):
            self.loaded_state = state_dict
            return None

    fake_nemo = types.SimpleNamespace(models=types.SimpleNamespace(ASRModel=FakeASRModel))
    fake_nemo_collections = types.ModuleType("nemo.collections")
    fake_nemo_collections.asr = fake_nemo
    fake_nemo_root = types.ModuleType("nemo")
    fake_nemo_root.collections = fake_nemo_collections

    with patch.dict(sys.modules, {"nemo": fake_nemo_root, "nemo.collections": fake_nemo_collections, "nemo.collections.asr": fake_nemo}):
        wrapper = module.nemotron_asr_model_loader(str(model_dir))

    assert isinstance(wrapper, module.NemotronASRWrapper)
    assert set(wrapper.model.loaded_state.keys()) == set(expected_state.keys())
    assert torch.equal(wrapper.model.loaded_state["encoder.weight"], expected_state["encoder.weight"])


def test_restore_quantized_state_dict_rebinds_quanttensor_parameters():
    module = _load_nemotron_module()

    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = nn.Linear(4, 3)

    model = NestedModel()
    weight = QuantTensor.from_float(torch.randn(3, 4), bits=4, symmetric=False, group_size=2)
    state_dict = {"sub.weight": weight, "sub.bias": torch.randn(3)}

    module._restore_quantized_state_dict(model, state_dict)

    assert isinstance(model.sub.weight, QuantTensor)
    assert model.sub.weight.qweight is not None
    assert torch.equal(model.sub.bias, state_dict["sub.bias"])
