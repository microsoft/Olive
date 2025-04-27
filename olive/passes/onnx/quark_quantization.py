#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import onnx

from olive.common.config_utils import validate_config
from olive.common.utils import exclude_keys, hash_string
from olive.data.config import DataConfig
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    get_external_data_config,
    model_has_adapters,
    model_proto_to_file,
    model_proto_to_olive_model,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.resource_path import LocalFile
from olive.search.search_parameter import Boolean, Categorical, Conditional

logger = logging.getLogger(__name__)

# pylint: disable=consider-using-with, attribute-defined-outside-init


# common config for Vitis-AI quantization
quark_quantization_config = {
    "config_template": PassConfigParam(
        type_=Optional[str],
        default_value=None,
        required=True,
        description="Quark configuration template to apply in quantization.",
    ),
    "data_config": PassConfigParam(
        type_=Optional[Union[DataConfig, Dict]],
        default_value=None,
        required=True,
        description="Data config for calibration.",
    ),
    "weight_type": PassConfigParam(
        type_=Optional[str],
        default_value=None,
        search_defaults=Categorical(["QInt8"]),
        description="""
            Data type for quantizing weights which is used in Quark quantization.
            'QInt8' for signed 8-bit integer,
        """,
    ),
    "input_nodes": PassConfigParam(
        type_=Optional[list],
        default_value=None,
        description="List of input nodes to be quantized. Default is an empty list.",
    ),
    "output_nodes": PassConfigParam(
        type_=Optional[list],
        default_value=None,
        description="List of output nodes to be quantized. Default is an empty list.",
    ),
    "op_types_to_quantize": PassConfigParam(
        type_=Optional[list],
        default_value=None,
        description="List of operation types to be quantized. Default is an empty list.",
    ),
    "extra_op_types_to_quantize": PassConfigParam(
        type_=Optional[list],
        default_value=None,
        description="List of additional operation types to be quantized. Default is an empty list.",
    ),
    "nodes_to_quantize": PassConfigParam(
        type_=Optional[list],
        default_value=None,
        description="List of node names to be quantized. Default is an empty list.",
    ),
    "nodes_to_exclude": PassConfigParam(
        type_=Optional[list],
        default_value=None,
        description="List of node names to be excluded from quantization. Default is an empty list.",
    ),
    "subgraphs_to_exclude": PassConfigParam(
        type_=Optional[list],
        default_value=None,
        description="""
            List of start and end node names of subgraphs to be excluded from quantization.
            Default is an empty list.
        """,
    ),
    "specific_tensor_precision": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to enable specific tensor precision. Default is False.",
    ),
    "execution_providers": PassConfigParam(
        type_=Optional[list],
        default_value=None,
        description="List of execution providers. Default is ['CPUExecutionProvider'].",
    ),
    "per_channel": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        search_defaults=Boolean(),
        description="Flag to enable per-channel quantization. Default is False.",
    ),
    "reduce_range": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to reduce quantization range. Default is False.",
    ),
    "optimize_model": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        search_defaults=Boolean(),
        description="Flag to optimize the model. Default is True.",
    ),
    "use_dynamic_quant": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to use dynamic quantization. Default is False.",
    ),
    "use_external_data_format": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to use external data format. Default is False.",
    ),
    "convert_fp16_to_fp32": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to convert FP16 to FP32. Default is False.",
    ),
    "convert_nchw_to_nhwc": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to convert NCHW to NHWC. Default is False.",
    ),
    "include_sq": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to include square root in quantization. Default is False.",
    ),
    "include_cle": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to include CLE in quantization. Default is False.",
    ),
    "include_auto_mp": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to include automatic mixed precision. Default is False.",
    ),
    "include_fast_ft": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to include fast fine-tuning. Default is False.",
    ),
    "quant_preprocess": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        search_defaults=Boolean(),
        description="""
            Shape inference and model optimization, in preparation for quantization.
            https://onnxruntime.ai/docs/performance/quantization.html#pre-processing
        """,
    ),
    "calibrate_method": PassConfigParam(
        type_=Optional[str],
        default_value=None,
        search_defaults=Categorical(["NonOverflow", "MinMSE"]),
        description="""
            Method used for calibration. Default is CalibrationMethod.MinMax.
        """,
    ),
    "quant_format": PassConfigParam(
        type_=Optional[str],
        default_value=None,
        search_defaults=Categorical(["QDQ", "QOperator"]),
        description="""
            Format of quantization. Default is QuantFormat.QDQ.
        """,
    ),
    "activation_type": PassConfigParam(
        type_=Optional[str],
        default_value=None,
        search_defaults=Conditional(
            parents=("quant_format", "weight_type"),
            support={
                ("QDQ", "QInt8"): Categorical(["QInt8"]),
                ("QDQ", "QUInt8"): Categorical(["QUInt8"]),
                ("QOperator", "QUInt8"): Categorical(["QUInt8"]),
                ("QOperator", "QInt8"): Conditional.get_invalid_choice(),
            },
        ),
        description="""
            Type of quantization for activations. Default is QuantType.QInt8.
        """,
    ),
    "enable_npu_cnn": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        search_defaults=Boolean(),
        description="Flag to enable NPU CNN. Default is False.",
    ),
    "enable_npu_transformer": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        search_defaults=Boolean(),
        description="Flag to enable NPU Transformer. Default is False.",
    ),
    "debug_mode": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to enable debug mode. Default is False.",
    ),
    "print_summary": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to print summary of quantization. Default is True.",
    ),
    "ignore_warnings": PassConfigParam(
        type_=Optional[bool],
        default_value=None,
        description="Flag to suppress the warnings globally. Default is True.",
    ),
    "log_severity_level": PassConfigParam(
        type_=Optional[int],
        default_value=None,
        description="0:DEBUG, 1:INFO, 2:WARNING. 3:ERROR, 4:CRITICAL/FATAL. Default is 1.",
    ),
}

_extra_options_config = {
    "extra_options": PassConfigParam(
        type_=dict,
        default_value=None,
        description="""
            Key value pair dictionary for `extra_options` in quantization.
        """,
    ),
}


class QuarkQuantization(Pass):
    """Quantize ONNX model with onnxruntime.

    We can search for best parameters for Quark quantization at same time.
    """

    def _initialize(self):
        super()._initialize()
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_quark_tmp")

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Override this method to return False by using the accelerator spec information."""
        return False

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "quant_mode": PassConfigParam(
                type_=str,
                default_value="static",
                search_defaults=Categorical(["static"]),
                description="""
                    Onnx Quantization mode.
                    'static' for vitis ai quantization.
                """,
            ),
            # common quantization config
            **deepcopy(quark_quantization_config),
            # exposed extra options config
            **deepcopy(_extra_options_config),
            # external data config
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if model_has_adapters(model.model_path):
            logger.info("Model has adapters which should not be quantized. Returning the model without quantization.")
            return model
        from quark.onnx.quantize import PowerOfTwoMethod, QuantFormat, QuantType

        # start with a copy of the config
        run_config = config.dict()

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # extra config
        extra_options = deepcopy(config.extra_options) if config.extra_options else {}

        # preprocess the model
        # we hash the entire path of the input model to ensure we are not accidentally using a preprocessed model
        # from a different model
        preprocessed_temp_model_path = (
            Path(self.tmp_dir.name) / f"{hash_string(str(Path(model.model_path).resolve()))[:8]}" / "preprocessed.onnx"
        )
        preprocessed_temp_model_path.parent.mkdir(exist_ok=True, parents=True)
        if run_config["quant_preprocess"]:
            if not preprocessed_temp_model_path.exists():
                # overwrite the model path with the preprocessed model path
                logger.info("Preprocessing model for quantization")
                model = self._quant_preprocess(model, preprocessed_temp_model_path)
            else:
                logger.info("Already processed model for quantization, skipping preprocessing")
                model = ONNXModelHandler(LocalFile({"path": preprocessed_temp_model_path}))

        config_template = run_config["config_template"]
        # keys not needed for quantization
        to_delete = [
            "data_config",
            "quant_mode",
            "quant_preprocess",
            "config_template",
        ]
        to_delete += list(get_external_data_config().keys())

        # update string values to enum values
        def map_to_enum(enum, value):
            return enum[value] if value is not None else None

        run_config.update(
            {
                "calibrate_method": map_to_enum(PowerOfTwoMethod, run_config["calibrate_method"]),
                "quant_format": map_to_enum(QuantFormat, run_config["quant_format"]),
                "activation_type": map_to_enum(QuantType, run_config["activation_type"]),
                "weight_type": map_to_enum(QuantType, run_config["weight_type"]),
                "extra_options": extra_options,
            }
        )

        # remove keys not needed for quantization
        run_config = exclude_keys(run_config, to_delete)

        # to be safe, run the quantizer with use_external_data_format set to `True` and
        # `model_output` to a temporary directory
        # reload the model and save to output_model_path using the external data config
        # TODO(XiaoSheng): don't default to use_external_data_format=True if the loading and saving model makes
        # the pass inefficient
        tmp_dir = tempfile.TemporaryDirectory(prefix="olive_quark_tmp")
        tmp_dir_path = Path(tmp_dir.name)
        tmp_model_path = str(tmp_dir_path / Path(output_model_path).name)

        # get the dataloader
        dataloader = None
        if config.data_config:
            data_config = validate_config(config.data_config, DataConfig)
            dataloader = data_config.to_data_container().create_calibration_dataloader()

        from quark.onnx import ModelQuantizer
        from quark.onnx.quantization.config import Config, get_default_config

        quant_config = get_default_config(config_template)
        for k, v in run_config.items():
            if k == "extra_options":
                quant_config.extra_options.update(v)
            elif v is None:
                continue
            else:
                setattr(quant_config, k, v)
        quantizer = ModelQuantizer(Config(global_quant_config=quant_config))
        quantizer.quantize_model(model.model_path, tmp_model_path, dataloader)

        # load the model
        onnx_model = onnx.load(tmp_model_path)
        # the model is loaded into memory, so it's safe to delete previously exported files
        tmp_dir.cleanup()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    def _quant_preprocess(self, model: ONNXModelHandler, output_model_path: str) -> ONNXModelHandler:
        from onnxruntime.quantization.preprocess import quant_pre_process

        try:
            quant_pre_process(
                input_model_path=model.model_path,
                output_model_path=str(output_model_path),
                auto_merge=True,
                save_as_external_data=True,
            )
        except Exception as e:
            # TODO(xiaosheng): try with `skip_optimization = True`
            # quantization preprocessing will fail if the model is too large and `skip_optimization = False`
            # there are some problems with the path to where the external data is saved
            # need to find out why before enabling this

            logger.warning("Failed to run quantization preprocessing with error of %s. Using original model.", e)
            # save original model to output path
            onnx_model = onnx.load(model.model_path)
            model_proto_to_file(
                onnx_model,
                output_model_path,
                save_as_external_data=True,  # always save as external data to avoid failures due to large models
            )

        # since this is only used internally, we will just treat it as a model file
        return ONNXModelHandler(LocalFile({"path": output_model_path}))
