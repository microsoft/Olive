# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path
from typing import Union

from olive.constants import ModelFileFormat
from olive.model import HfModelHandler, PyTorchModelHandler2


def inherit_hf_from_hf(
    model: HfModelHandler, model_path: Union[str, Path], adapter_path: Union[str, Path] = None
) -> HfModelHandler:
    model_config = model.to_json()["config"]
    model_config["model_path"] = model_path
    model_config["adapter_path"] = adapter_path
    return HfModelHandler(**model_config)


def inherit_pytorch_from_hf(
    model: HfModelHandler,
    model_path: Union[str, Path],
    model_file_format: ModelFileFormat = ModelFileFormat.PYTORCH_ENTIRE_MODEL,
    **extra_attributes,
) -> PyTorchModelHandler2:
    return PyTorchModelHandler2(
        model_path=model_path,
        model_file_format=model_file_format,
        io_config=deepcopy(model.io_config),
        model_attributes={**model.model_attributes, **extra_attributes},
        generative=model.generative,
    )


def inherit_pytorch_from_pytorch(model: PyTorchModelHandler2, model_path: Union[str, Path]) -> PyTorchModelHandler2:
    model_config = model.to_json()["config"]
    model_config["model_path"] = model_path
    model_config.pop("model_loader", None)
    return PyTorchModelHandler2(**model_config)
