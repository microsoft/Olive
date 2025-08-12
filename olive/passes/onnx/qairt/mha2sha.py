# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtMHA2SHA(Pass):
    """Runs QAIRT MHA to SHA transformation on ONNX model splits and saves the transformed models.

    Uses transformation API from the QAIRT SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "mha2sha_kwargs": PassConfigParam(
                type_=dict[str, Any],
                default_value=None,
                description="Additional parameters to be passed to the MHA2SHA transformation function.",
            ),
        }

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise NotImplementedError(
                f"QairtMHA2SHA pass only supports ONNXModelHandler, but received type {type(model)}"
            )

        try:
            from qti.aisw.tools.core.utilities.framework.frameworks.onnx import OnnxModel
        except ImportError:
            try:
                # Backwards compatibility with older locations of OnnxModel in <= QAIRT 2.36.1
                from qti.aisw.tools.core.utilities.framework.onnx import OnnxModel
            except ImportError as e:
                raise ImportError("Please install qti.aisw.tools and all dependencies to use QairtMHA2SHA.") from e

        qairt_onnx_model = OnnxModel.load(model_path=model.model_path)
        try:
            qairt_onnx_model.mha2sha_v2(**(config.mha2sha_kwargs if config.mha2sha_kwargs is not None else {}))
        except AttributeError:
            # Backwards compatibility with older definitions of OnnxModel in <= QAIRT 2.37
            logger.warning("MHA2SHA V2 is not available for this SDK version, defaulting to MHA2SHA V1")
            qairt_onnx_model.mha2sha(**(config.mha2sha_kwargs if config.mha2sha_kwargs is not None else {}))

        qairt_onnx_model.export(output_model_path, prefix=Path(model.model_path).stem)

        return ONNXModelHandler(
            model_path=output_model_path,
            onnx_file_name=model.onnx_file_name,
            model_attributes=deepcopy(model.model_attributes),
        )
