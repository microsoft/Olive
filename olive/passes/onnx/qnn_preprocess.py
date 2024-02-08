# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Any, Callable, Dict

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam


class QNNPreprocess(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "fuse_layernorm": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=("Whether to fuse ReduceMean sequence into a single LayerNormalization node."),
            )
        }

    @staticmethod
    def _validators() -> Dict[str, Callable[..., Any]]:
        pass

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
    ) -> ONNXModelHandler:
        from onnxruntime import __version__ as OrtVersion
        from packaging import version

        if version.parse(OrtVersion) < version.parse("1.17.0"):
            raise RuntimeError("QNNPreprocess only supports ONNXRuntime version 1.17.0 or later")

        from onnxruntime.quantization.execution_providers.qnn import qnn_preprocess_model

        modified = qnn_preprocess_model(model.model_path, output_model_path, **config)
        if not modified:
            return model
        return ONNXModelHandler(output_model_path)
