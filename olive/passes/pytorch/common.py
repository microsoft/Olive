# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path
from typing import Union

from olive.model import HfModelHandler, PyTorchModelHandler2


def inherit_pytorch_from_hf(model: HfModelHandler, new_model_path: Union[str, Path]) -> PyTorchModelHandler2:
    return PyTorchModelHandler2(
        model_path=new_model_path,
        io_config=deepcopy(model.io_config),
        model_attributes=deepcopy(model.model_attributes),
        generative=model.generative,
    )


def inherit_pytorch_from_pytorch(model: PyTorchModelHandler2, new_model_path: Union[str, Path]) -> PyTorchModelHandler2:
    model_config = model.to_json()["config"]
    model_config["model_path"] = new_model_path
    model_config.pop("model_loader", None)
    return PyTorchModelHandler2(**model_config)
