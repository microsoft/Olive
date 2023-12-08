# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import List, Tuple

from olive.constants import ModelFileFormat
from olive.model.config.registry import model_handler_registry
from olive.model.handler.pytorch import PyTorchModelHandler


@model_handler_registry("OptimumModel")
class OptimumModelHandler(PyTorchModelHandler):
    """TODO(myguo): need refactor this class to support Optimum model."""

    jsonify_config_keys: Tuple[str, ...] = ("model_components",)

    def __init__(self, model_components: List[str], **kwargs):
        kwargs = kwargs or {}
        kwargs["model_file_format"] = ModelFileFormat.COMPOSITE_MODEL
        super().__init__(**kwargs)
        self.model_components = model_components
