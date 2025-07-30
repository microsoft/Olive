# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

import onnx
from packaging import version

from olive.common.config_utils import ParamCategory, validate_config
from olive.common.utils import StrEnumBase
from olive.constants import Precision
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    get_external_data_config,
    model_has_adapters,
    model_proto_to_olive_model,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.search.search_parameter import Categorical

logger = logging.getLogger(__name__)

# pylint: disable=consider-using-with


def precision_to_qtype(p: Precision):
    """Convert precision to aimet qtype."""
    from aimet_onnx import float16, int4, int8, int16

    precision_mapping = {
        Precision.INT4: int4,
        Precision.INT8: int8,
        Precision.INT16: int16,
        Precision.UINT4: int4,
        Precision.UINT8: int8,
        Precision.UINT16: int16,
        Precision.FP16: float16,
    }
    return precision_mapping.get(p)


class QuantScheme(StrEnumBase):
    MIN_MAX = "min_max"
    TF_ENHANCED = "tf_enhanced"


def _has_quantization_nodes(model: onnx.ModelProto):
    quantize_op_types = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "MatMulNBits"}
    return any(node.op_type in quantize_op_types for node in model.graph.node)


class AimetQuantization(Pass):
    """Quantize ONNX model using aimet-onnx."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "precision": PassConfigParam(
                type_=Precision,
                default_value=Precision.INT8,
                search_defaults=Categorical([Precision.INT4, Precision.INT8, Precision.INT16]),
                description="Quantization precision for quantizing weights.",
            ),
            "activation_type": PassConfigParam(
                type_=Precision,
                default_value=Precision.UINT8,
                search_defaults=Categorical([Precision.UINT8, Precision.UINT16, Precision.FP16]),
                description="Quantization precision for quantizing activations.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                required=True,
                description="Data config for calibration.",
            ),
            "quant_scheme": PassConfigParam(
                type_=QuantScheme,
                default_value="min_max",
                search_defaults=Categorical(["min_max", "tf_enhanced"]),
                description="Quantization scheme to use for calibration. Current methods supported are min_max and tfe.",
            ),
            "config_file": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                required=False,
                category=ParamCategory.PATH,
                description="Path to AIMET config file defining target hardware quantization support.",
            ),
            "calibration_providers": PassConfigParam(
                type_=list,
                default_value=None,
                description="""
                    Execution providers to run the session during calibration.
                    Supported providers are {"CUDAExecutionProvider", "CPUExecutionProvider"}
                    Default is None which uses [ "CPUExecutionProvider" ].
                """,
            ),
        }
        config.update(get_external_data_config())
        return config

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.precision not in (Precision.INT4, Precision.INT8, Precision.INT16):
            logger.warning("Unsupported param_type: %s", config.precision)
            return False

        if config.activation_type not in (Precision.UINT8, Precision.UINT16, Precision.FP16):
            logger.warning("Unsupported activation_type: %s", config.activation_type)
            return False

        if config.quant_scheme not in ("min_max", "tf_enhanced"):
            logger.warning("Unsupported quant_scheme: %s", config.quant_scheme)
            return False

        return True

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if model_has_adapters(model.model_path):
            logger.info("Model has adapters which should not be quantized. Returning the model without quantization.")
            return model

        import aimet_onnx
        from onnxruntime import __version__ as OrtVersion

        if version.parse(OrtVersion) < version.parse("1.19.0"):
            raise ValueError("AIMET Quantization is only supported for onnxruntime>=1.19.0")

        run_config = config.dict()
        param_type = precision_to_qtype(run_config.get("precision"))
        act_type = precision_to_qtype(run_config.get("activation_type"))

        if not param_type:
            raise ValueError(f"Unsupported precision: {run_config.get('precision')}")

        if not act_type:
            raise ValueError(f"Unsupported activation_type: {run_config.get('activation_type')}")

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        data_config = validate_config(config.data_config, DataConfig)

        # Note: bool(_CalibrationDataReader) is not implemented, convert to generator to avoid error
        calib_dataloader = (x for x in data_config.to_data_container().create_calibration_dataloader())

        onnx_model = onnx.load(model.model_path)

        if _has_quantization_nodes(onnx_model):
            raise NotImplementedError("AIMET Quantization does not support pre-quantized models")

        with tempfile.TemporaryDirectory(prefix="olive_tmp") as tmp_dir:
            sim = aimet_onnx.QuantizationSimModel(
                onnx_model,
                param_type=param_type,
                activation_type=act_type,
                config_file=run_config.get("config_file"),
                quant_scheme=run_config.get("quant_scheme", "min_max"),
                providers=run_config.get("calibration_providers"),
                path=tmp_dir,
            )
            sim.compute_encodings(calib_dataloader)
            qdq_model = sim.to_onnx_qdq()

        return model_proto_to_olive_model(qdq_model, output_model_path, config)
