# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict

import torch

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModelHandler
from olive.model.utils.hf_utils import save_hf_model_tokenizer
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class MergeAdapterWeights(Pass):
    """Merge adapter weights into the base model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {}

    @torch.no_grad()
    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        if not model.adapter_path:
            raise RuntimeError(
                "No adapter path found in the model. Please check your input "
                "model type or remove `MergeAdapterWeights` from passes configs"
            )
        pytorch_model = model.load_model()
        merged_model = pytorch_model.merge_and_unload()

        merged_model.save_pretrained(output_model_path)
        save_hf_model_tokenizer(
            model.get_hf_model_tokenizer(), output_model_path, **model.hf_config.get_loading_args_from_pretrained()
        )

        return PyTorchModelHandler(
            output_model_path,
            model_attributes=model.model_attributes,
            model_file_format=model.model_file_format,
            io_config=model.io_config,
            hf_config=model.hf_config,
            adapter_path=None,
            mlflow_transformer_model_cache_dir=model.mlflow_transformer_model_cache_dir,
        )
