# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from typing import Any, Dict

import torch

from olive.common.utils import get_attr
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModel
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam


class DeQuantizeHF(Pass):
    """Dequantize a Hugging Face PyTorch model.

    This pass only supports PyTorchModel with hf_config.
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {}

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModel:
        import bitsandbytes as bnb

        # check that model has hf_config
        # TODO(jambayk): support models without hf_config if needed, need to consider how the model is saved and loaded
        if not model.hf_config:
            raise ValueError("DeQuantizeHF pass only supports PyTorchModel with hf_config.")

        # don't want the original loaded model
        # also frees gpu memory if original model is on gpu
        model.model = None
        # create copy of the input model, will modify this model
        new_model = deepcopy(model)
        # don't want to load adapters
        new_model.set_resource("adapter_path", None)
        # load model
        pytorch_model = new_model.load_model()

        # TODO(jambayk): support other quantization schemes if needed
        if pytorch_model.is_loaded_in_8bit:
            raise ValueError("Model is quantized to 8-bit but DeQuantizeHF only supports 4-bit quantization.")

        # dequantize model
        # based on https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
        for name, module in pytorch_model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                weight = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state)
                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=weight.dtype)
                new_module.weight.data = weight
                # move to cpu to save gpu memory
                new_module.to(device="cpu", dtype=pytorch_model.dtype)

                # replace module
                parent_name = ".".join(name.split(".")[:-1])
                parent = get_attr(pytorch_model, parent_name)
                module_name = name.split(".")[-1]
                setattr(parent, module_name, new_module)

        # reset config
        pytorch_model.is_loaded_in_4bit = False
        if hasattr(pytorch_model.config, "quantization_config"):
            del pytorch_model.config.quantization_config
        if hasattr(pytorch_model.config, "pretraining_tp"):
            del pytorch_model.config.pretraining_tp

        # save new weights
        pytorch_model.save_pretrained(output_model_path)

        # prepare new model
        new_model.model = None
        new_model.set_resource("model_path", output_model_path)
        new_model.set_resource("adapter_path", model.get_resource("adapter_path"))
        if new_model.hf_config.model_loading_args:
            new_model.hf_config.model_loading_args.quantization_method = None
            new_model.hf_config.model_loading_args.quantization_config = None
        # TODO(jambayk): what about the torch_dtype? should it be set to None?

        return new_model
