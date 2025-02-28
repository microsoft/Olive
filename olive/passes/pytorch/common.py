# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path
from typing import Union

from olive.constants import ModelFileFormat
from olive.model import DistributedHfModelHandler, HfModelHandler, PyTorchModelHandler


def inherit_hf_from_hf(
    model: HfModelHandler, model_path: Union[str, Path], adapter_path: Union[str, Path] = None, load_kwargs: dict = None
) -> HfModelHandler:
    model_config = model.to_json()["config"]
    model_config["model_path"] = model_path
    model_config["adapter_path"] = adapter_path
    if load_kwargs:
        # only overwrite the load_kwargs if provided
        model_config["load_kwargs"] = load_kwargs
    return HfModelHandler(**model_config)


def inherit_distributed_hf_from_hf(
    model: HfModelHandler, model_path: Union[str, Path], model_name_pattern: str, num_ranks: int
) -> HfModelHandler:
    model_config = model.to_json()["config"]
    model_config["model_path"] = model_path
    del model_config["adapter_path"]
    model_config.update(
        {
            "model_name_pattern": model_name_pattern,
            "num_ranks": num_ranks,
        }
    )
    return DistributedHfModelHandler(**model_config)


def inherit_pytorch_from_hf(
    model: HfModelHandler,
    model_path: Union[str, Path],
    model_file_format: ModelFileFormat = ModelFileFormat.PYTORCH_ENTIRE_MODEL,
    **extra_attributes,
) -> PyTorchModelHandler:
    # keep original io_config if present otherwise use the hf onnx_config
    io_config = model.to_json()["config"].get("io_config")
    if not io_config and model.io_config:
        # must have come from the automatic io config using optimum
        hf_io_config = deepcopy(model.io_config)
        hf_dummy_inputs = model.get_dummy_inputs()

        dynamic_shapes = hf_io_config.get("dynamic_shapes", {})
        if isinstance(dynamic_shapes, dict):
            {
                k: v
                for k, v in hf_io_config.get("dynamic_axes", {}).items()
                if not k.startswith(("present", "past_key_values"))
            }

        # kv cache will be handled by the kv_cache flag in io_config
        io_config = {
            "input_names": [i for i in hf_io_config.get("input_names", []) if not i.startswith("past_key_values")],
            "input_shapes": [],
            "input_types": [],
            "output_names": [o for o in hf_io_config.get("output_names", []) if not o.startswith("present")],
            "dynamic_axes": {
                k: v
                for k, v in hf_io_config.get("dynamic_axes", {}).items()
                if not k.startswith(("present", "past_key_values"))
            },
            "dynamic_shapes": dynamic_shapes,
        }

        for i_name in io_config["input_names"]:
            io_config["input_shapes"].append(hf_dummy_inputs[i_name].shape)
            io_config["input_types"].append(str(hf_dummy_inputs[i_name].dtype).strip("torch."))

    if io_config and not io_config.get("kv_cache") and model.task.endswith("-with-past"):
        io_config["kv_cache"] = True

    # dynamic_shapes deals with kv_cache here. If kv_cache is False,
    # we remove past_key_values from dynamic_shapes
    if not io_config.get("kv_cache", False):
        dynamic_shapes = {
            k: v for k, v in hf_io_config.get("dynamic_shapes", {}).items() if not k.startswith("past_key_values")
        }
    else:
        dynamic_shapes = hf_io_config.get("dynamic_shapes", {})
    io_config["dynamic_shapes"] = dynamic_shapes

    return PyTorchModelHandler(
        model_path=model_path,
        model_file_format=model_file_format,
        io_config=io_config,
        model_attributes={**model.model_attributes, **extra_attributes},
        generative=model.generative,
    )


def inherit_pytorch_from_pytorch(model: PyTorchModelHandler, model_path: Union[str, Path]) -> PyTorchModelHandler:
    model_config = model.to_json()["config"]
    model_config["model_path"] = model_path
    model_config.pop("model_loader", None)
    return PyTorchModelHandler(**model_config)
