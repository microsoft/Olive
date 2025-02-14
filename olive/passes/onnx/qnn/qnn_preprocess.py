# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict, Type

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes.olive_pass import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QNNPreprocess(Pass):
    """Preprocess ONNX model for quantization targeting QNN Execution Provider."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "fuse_layernorm": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description="Whether to fuse ReduceMean sequence into a single LayerNormalization node.",
            ),
            "inputs_to_make_channel_last": PassConfigParam(
                type_=list,
                default_value=None,
                required=False,
                description="""inputs_to_make_channel_last: List of graph input names to transpose to be
                "channel-last". For example, if "input0" originally has the shape (N, C, D1, D2, ..., Dn),
                the resulting model will change input0's shape to (N, D1, D2, ..., Dn, C) and add a transpose
                node after it.

                Original:
                    input0 (N, C, D1, D2, ..., Dn) --> <Nodes>

                Updated:
                    input0 (N, D1, D2, ..., Dn, C) --> Transpose
                        --> input0_chanfirst (N, C, D1, D2, ..., Dn) --> <Nodes>

                This can potentially improve inference latency for QDQ models running on QNN EP because the
                additional transpose node may allow other transpose nodes inserted during ORT layout
                transformation to cancel out.""",
            ),
            "outputs_to_make_channel_last": PassConfigParam(
                type_=list,
                default_value=None,
                required=False,
                description="""List of graph output names to transpose to be "channel-last". For example,
            if "output0" originally has the shape (N, C, D1, D2, ..., Dn), the resulting model will change
            output0's shape to (N, D1, D2, ..., Dn, C) and add a transpose node before it.

            Original:
                <Nodes> --> output0 (N, C, D1, D2, ..., Dn)

            Updated:
                <Nodes> --> output0_chanfirst (N, C, D1, D2, ..., Dn) --> Transpose
                    --> output0 (N, D1, D2, ..., Dn, C)

            This can potentially improve inference latency for QDQ models running on QNN EP because the
            additional transpose node may allow other transpose nodes inserted during ORT layout transformation
            to cancel out.""",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        from onnxruntime import __version__ as OrtVersion
        from packaging import version

        if version.parse(OrtVersion) < version.parse("1.17.0"):
            raise RuntimeError("QNNPreprocess only supports ONNXRuntime version 1.17.0 or later")

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        external_data_location = config.external_data_name or f"{Path(output_model_path).name}.data"

        # only 1.18.0 or later adds the following parameters
        extra_kwargs = {
            "save_as_external_data": config.save_as_external_data,
            "all_tensors_to_one_file": config.all_tensors_to_one_file,
            "external_data_size_threshold": config.size_threshold,
            "external_data_location": external_data_location,
            "external_data_convert_attribute": config.convert_attribute,
            "inputs_to_make_channel_last": config.inputs_to_make_channel_last,
            "outputs_to_make_channel_last": config.outputs_to_make_channel_last,
        }
        if version.parse(OrtVersion) < version.parse("1.18.0"):
            removed_config = [
                "inputs_to_make_channel_last",
                "outputs_to_make_channel_last",
                *get_external_data_config(),
            ]
            logger.info(
                "Following config settings will be ignored as they are not supported in ONNXRuntime < 1.18.0: %s",
                ", ".join(removed_config),
            )
            extra_kwargs = {}

        from onnxruntime.quantization.execution_providers.qnn import qnn_preprocess_model

        modified = qnn_preprocess_model(
            model.model_path,
            output_model_path,
            fuse_layernorm=config.fuse_layernorm,
            **extra_kwargs,
        )
        if not modified:
            return model
        return ONNXModelHandler(output_model_path)
