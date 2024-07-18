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
    model: HfModelHandler, model_path: Union[str, Path], adapter_path: Union[str, Path] = None
) -> HfModelHandler:
    model_config = model.to_json()["config"]
    model_config["model_path"] = model_path
    model_config["adapter_path"] = adapter_path
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
    if io_config and not io_config.get("kv_cache") and model.task.endswith("-with-past"):
        io_config["kv_cache"] = True
    elif model.io_config:
        io_config = deepcopy(model.io_config)

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
