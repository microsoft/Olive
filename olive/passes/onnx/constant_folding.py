# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam


class GSConstantFolding(Pass):
    """Uses onnx-graphsurgeon to fold constants in the model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        import onnx_graphsurgeon as gs
        from onnxruntime.transformers.onnx_model import OnnxModel

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        onnx_model = OnnxModel(model.load_model())
        onnx_model.remove_useless_cast_nodes()

        graph = gs.import_onnx(onnx_model.model)
        graph.fold_constants().cleanup()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(gs.export_onnx(graph), output_model_path, config)
