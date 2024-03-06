# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Callable, Dict

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes.olive_pass import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import PassConfigParam


class QNNPreprocess(Pass):
    """Preprocess ONNX model for quantization targeting QNN Execution Provider."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "fuse_layernorm": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description=("Whether to fuse ReduceMean sequence into a single LayerNormalization node."),
            )
        }
        # only 1.18.0 or later adds the following parameters
        from onnxruntime import __version__ as OrtVersion
        from packaging import version

        if version.parse(OrtVersion) > version.parse("1.18.0"):
            config.update(get_external_data_config())
        return config

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

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        modified = qnn_preprocess_model(model.model_path, output_model_path, **config)
        if not modified:
            return model
        return ONNXModelHandler(output_model_path)
