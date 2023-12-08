# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import List

from olive.common.config_utils import serialize_to_json
from olive.constants import ModelFileFormat
from olive.model.config.registry import model_handler_registry
from olive.model.handler.pytorch import PyTorchModelHandler


@model_handler_registry("OptimumModel")
class OptimumModelHandler(PyTorchModelHandler):
    """TODO(myguo): need refactor this class to support Optimum model."""

    def __init__(self, model_components: List[str], **kwargs):
        kwargs = kwargs or {}
        kwargs["model_file_format"] = ModelFileFormat.COMPOSITE_MODEL
        super().__init__(**kwargs)
        self.model_components = model_components

    def to_json(self, check_object: bool = False):
        config = super().to_json(check_object)
        config["config"].update({"model_components": self.model_components})
        return serialize_to_json(config, check_object)
