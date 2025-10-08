# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

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


def _get_dataloader(data_config, model_path, io_config, providers):
    # Note: bool(_CalibrationDataReader) is not implemented, convert to generator to avoid error
    return (
        x
        for x in data_config.to_data_container().create_calibration_dataloader(
            model_path=model_path, io_config=io_config, calibration_providers=providers
        )
    )


class QuantScheme(StrEnumBase):
    MIN_MAX = "min_max"
    TF_ENHANCED = "tf_enhanced"


def _has_qdq_nodes(model: onnx.ModelProto):
    quantize_op_types = {"QuantizeLinear", "DequantizeLinear"}
    return any(node.op_type in quantize_op_types for node in model.graph.node)


def _has_dynamic_quantization(model: onnx.ModelProto):
    return any(node.op_type == "DynamicQuantizeLinear" for node in model.graph.node)


def _disable_quantizer(sim, tensor_name: str):
    quantizer = sim.qc_quantize_op_dict.get(tensor_name)
    if quantizer and not quantizer.is_encoding_frozen():
        quantizer.enabled = False


def _exclude_op_types(sim, op_types_to_exclude: list[str]):
    """Excludes tensors from quantization if they are inputs/outputs only to nodes with op_type in op_types_to_exclude.

    Quantizers will be disabled only for tensors which are:
     - Intermediate tensors produced by and consumed only by nodes with excluded op types
     - Model inputs that feed only to excluded op types
     - Model outputs produced by excluded op types
    """
    for product in sim.connected_graph.get_all_products().values():
        if product.producer and product.producer.type not in op_types_to_exclude:
            continue

        if any(consumer.type not in op_types_to_exclude for consumer in product.consumers):
            continue

        _disable_quantizer(sim, product.name)


def _apply_precision_overrides(sim, tensor_precision_overrides: dict[str, Precision]):
    for name, precision in tensor_precision_overrides.items():
        qtype = precision_to_qtype(precision)
        quantizer = sim.qc_quantize_op_dict.get(name)
        if not quantizer:
            raise RuntimeError(f"No quantizer found for tensor {name}")

        data_type, bits = qtype.to_legacy_repr()
        quantizer.data_type = data_type
        quantizer.set_bitwidth(bits)


SUPPORTED_TECHNIQUES: dict[str, "_AimetTechnique"] = {}


class _AimetTechnique:
    @staticmethod
    def apply(sim, **kwargs):
        raise NotImplementedError

    @classmethod
    def __init_subclass__(cls):
        SUPPORTED_TECHNIQUES[cls.__name__.lower()] = cls

    @classmethod
    def _requires_data(cls):
        signature = inspect.signature(cls.apply)
        return "data_config" in signature.parameters

    @classmethod
    def validate_args(cls, **kwargs):
        signature = inspect.signature(cls.apply)
        try:
            signature.bind(None, **kwargs)
            return True
        except TypeError as e:
            logger.warning("Unsupported arguments for technique %s:\n%s\n", cls.__qualname__, e)
            return False


class LPBQ(_AimetTechnique):
    @staticmethod
    def apply(  # pylint: disable=arguments-differ
        sim,
        *,
        op_types: Iterable[str] = ("Conv", "Gemm", "MatMul"),
        bitwidth: int = 4,
        decompressed_bw: int = 8,
        block_size: int = 64,
        nodes_to_exclude: Optional[list[str]] = None,
    ):
        from aimet_onnx.quantsim import set_grouped_blockwise_quantization_for_weights

        set_grouped_blockwise_quantization_for_weights(
            sim,
            op_types=op_types,
            bitwidth=bitwidth,
            decompressed_bw=decompressed_bw,
            block_size=block_size,
            excluded_nodes=nodes_to_exclude,
        )

        return sim


class Adaround(_AimetTechnique):
    @staticmethod
    def apply(  # pylint: disable=arguments-differ
        sim, *, data_config=None, num_iterations: int = 10000, nodes_to_exclude: Optional[list[str]] = None
    ):
        from aimet_onnx import apply_adaround
        from aimet_onnx.adaround.adaround_weight import AdaroundSupportedModules

        if nodes_to_exclude is not None:
            nodes_to_optimize = {
                node.name for node in sim.connected_graph.ordered_ops if node.type in AdaroundSupportedModules
            }
            nodes_to_optimize -= set(nodes_to_exclude)
        else:
            nodes_to_optimize = None

        apply_adaround(sim, list(data_config), num_iterations, nodes_to_optimize)

        return sim


class SeqMSE(_AimetTechnique):
    @staticmethod
    def apply(  # pylint: disable=arguments-differ
        sim,
        *,
        data_config=None,
        num_candidates: int = 20,
    ):
        """Apply aimet_onnx sequential MSE technique to sim.

        Args:
            sim: QuantizationSimModel to optimize.
            data_config: Dataset to use for optimization. If not specified for the technique, will default to the calibration data.
            num_candidates: Number of encoding candidates to sweep for each weight.

        """
        from aimet_onnx import apply_seq_mse

        apply_seq_mse(sim, data_config, num_candidates)

        return sim


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
            "op_types_to_exclude": PassConfigParam(
                type_=list[str],
                default_value=None,
                description="List of operator types to exclude from quantization.",
            ),
            "techniques": PassConfigParam(
                type_=list[dict[str, Any]],
                default_value=[],
                required=False,
                description="List of techniques to apply in order, each with its name and parameters",
            ),
            "tensor_precision_overrides": PassConfigParam(
                type_=dict[str, Precision],
                default_value={},
                required=False,
                description="Dictionary of tensor name to quantization precision.",
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

        for technique in config.techniques:
            if "name" not in technique:
                logger.warning("Techniques must specify a name")
                return False

            name = technique["name"].lower()
            technique_cls = SUPPORTED_TECHNIQUES.get(name)
            if not technique_cls:
                logger.warning("Unsupported technique: %s", name)
                return False

            if not technique_cls.validate_args(**{key: value for key, value in technique.items() if key != "name"}):
                return False

        for name, precision in config.tensor_precision_overrides.items():
            if not precision_to_qtype(precision):
                logger.warning("Unsupported precision %s for tensor %s", precision, name)
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

        onnx_model = onnx.load(model.model_path)

        if _has_dynamic_quantization(onnx_model):
            raise NotImplementedError("AIMET Quantization does not support dynamically quantized models.")

        with tempfile.TemporaryDirectory(prefix="olive_tmp") as tmp_dir:
            # pylint:disable = protected-access
            sim_initializer = (
                aimet_onnx.QuantizationSimModel
                if not _has_qdq_nodes(onnx_model)
                else aimet_onnx.QuantizationSimModel._from_onnx_qdq
            )
            sim = sim_initializer(
                onnx_model,
                param_type=param_type,
                activation_type=act_type,
                config_file=run_config.get("config_file"),
                quant_scheme=run_config.get("quant_scheme", "min_max"),
                providers=run_config.get("calibration_providers"),
                path=tmp_dir,
            )

            _apply_precision_overrides(sim, run_config["tensor_precision_overrides"])

            op_types_to_exclude = run_config["op_types_to_exclude"]
            if op_types_to_exclude:
                _exclude_op_types(sim, op_types_to_exclude)

            techniques = run_config["techniques"]
            for technique in techniques:
                name = technique.pop("name").lower()
                technique_impl = SUPPORTED_TECHNIQUES[name]

                if technique_impl._requires_data():
                    # If no data_config provided for technique, use calibration data
                    data_config = technique.get("data_config", None) or config.data_config
                    data_config = validate_config(data_config, DataConfig)
                    technique["data_config"] = _get_dataloader(
                        data_config, model.model_path, model.io_config, run_config["calibration_providers"]
                    )

                sim = technique_impl.apply(sim, **technique)

            data_config = validate_config(config.data_config, DataConfig)
            calib_dataloader = _get_dataloader(
                data_config, model.model_path, model.io_config, run_config["calibration_providers"]
            )

            sim.compute_encodings(calib_dataloader)
            qdq_model = sim.to_onnx_qdq(prequantize_constants=True)

        return model_proto_to_olive_model(qdq_model, output_model_path, config)
