"""Reload RNNT quantization artifacts across workflow process boundaries."""

import torch

from olive.common.quant.state_dict import install_quant_tensor_param
from olive.common.quant.tensor import QuantTensor
from olive.common.user_module_loader import UserModuleLoader
from olive.model.config import ModelConfig


def load_quantized_rnnt_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ModelConfig.from_json(checkpoint["model_config"]).create_model().load_model(cache_model=False)
    state_dict = checkpoint["state_dict"]
    quantized_weights = {name: value for name, value in state_dict.items() if isinstance(value, QuantTensor)}
    dense_state = {name: value for name, value in state_dict.items() if name not in quantized_weights}
    incompatible = model.load_state_dict(dense_state, strict=False)
    if set(incompatible.missing_keys) != set(quantized_weights) or incompatible.unexpected_keys:
        raise ValueError("Quantized RNNT checkpoint does not match the original model architecture.")
    for name, weight in quantized_weights.items():
        module_name, _, parameter_name = name.rpartition(".")
        install_quant_tensor_param(model.get_submodule(module_name), parameter_name, weight)
    return model


def _call_original_hook(model, hook_name):
    config = model.model_attributes["rnnt_input_model"]["config"]
    loader = UserModuleLoader(config.get("model_script"), config.get("script_dir"))
    return loader.call_object(config[hook_name], model)


def get_rnnt_dummy_inputs(model):
    return _call_original_hook(model, "dummy_inputs_func")


def get_rnnt_io_config(model):
    return _call_original_hook(model, "io_config")