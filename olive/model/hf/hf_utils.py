# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from itertools import chain
from pathlib import Path
from typing import Callable, Optional, Union

from olive.common.user_module_loader import UserModuleLoader
from olive.model.hf.hf_model import HFConfig, HFModelBase


def load_huggingface_model_from_task(task: str, name: str):
    """Load huggingface model from task and name"""
    from transformers import AutoModel
    from transformers.pipelines import check_task

    task_results = check_task(task)
    assert isinstance(task_results, tuple)
    if len(task_results) == 2:
        targeted_task = task_results[0]
    elif len(task_results) == 3:
        targeted_task = task_results[1]
    else:
        raise ValueError("unsupported transfomers version")

    model_class = {"pt": targeted_task["pt"]}
    class_tuple = ()
    class_tuple = class_tuple + model_class.get("pt", (AutoModel,))

    model = None
    for model_class in class_tuple:
        try:
            model = model_class.from_pretrained(name)
            return model
        except (OSError, ValueError):
            continue

    return model


def huggingface_model_loader(model_loader):
    import transformers

    if model_loader is None:
        model_loader = "AutoModel"
    if isinstance(model_loader, str):
        try:
            model_loader = getattr(transformers, model_loader)
        except AttributeError:
            raise AttributeError(f"{model_loader} is not found in transformers")
    elif not isinstance(model_loader, Callable):
        raise ValueError("model_loader must be a callable or a string defined in transformers")

    return model_loader.from_pretrained


def get_hf_model_config(model_name: str):
    """
    Get HF Config for the given model name
    """
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_name)


def load_huggingface_model_from_model_class(model_class: str, name: str):
    """
    Load huggingface model from model_loader and name
    """
    return huggingface_model_loader(model_class)(name)


def load_huggingface_model_from_custom_implementation(
    hf_config: HFConfig, model_script: Optional[Union[str, Path]] = None, script_dir: Optional[Union[str, Path]] = None
):
    user_module_loader = UserModuleLoader(model_script, script_dir)

    component_dict = {}
    for component in hf_config.components:
        component_dict[component.name] = user_module_loader.load_object(component.component_func)

    return HFModelBase(components=component_dict, config=hf_config.config)


def get_onnx_config(model_name: str, task: str, feature: str):
    from transformers.onnx import FeaturesManager

    model = load_huggingface_model_from_task(task, model_name)
    _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    return model_onnx_config(model.config)


def get_hf_model_io_config(model_name: str, task: str, feature: str):
    model_config = get_onnx_config(model_name, task, feature)
    inputs = model_config.inputs
    outputs = model_config.outputs
    io_config = {}
    io_config["input_names"] = list(inputs.keys())
    io_config["output_names"] = list(outputs.keys())
    io_config["dynamic_axes"] = dict(chain(inputs.items(), outputs.items()))
    return io_config


def get_hf_model_dummy_input(model_name: str, task: str, feature: str):
    from transformers import AutoTokenizer

    model_config = get_onnx_config(model_name, task, feature)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model_config.generate_dummy_inputs(tokenizer, framework="pt")
