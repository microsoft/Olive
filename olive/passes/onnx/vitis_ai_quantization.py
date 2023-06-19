#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, Union

from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

from olive.cache import get_local_path
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.vitis_ai import quantize_static
from olive.passes.onnx.vitis_ai.quant_utils import PowerOfTwoMethod
from olive.passes.pass_config import PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS
from olive.strategy.search_parameter import Boolean, Categorical, Conditional

logger = logging.getLogger(__name__)

# common config for vai_q_onnx quantization
vai_q_onnx_quantization_config = {
    "data_dir": PassConfigParam(
        type_=OLIVE_RESOURCE_ANNOTATIONS,
        is_path=True,
        description="""
            Path to the directory containing the dataset.
        """,
    ),
    "batch_size": PassConfigParam(
        type_=int,
        default_value=1,
        description="""
            Batch size for calibration, required.
        """,
    ),
    "dataloader_func": PassConfigParam(
        type_=Union[Callable, str],
        required=True,
        is_object=True,
        description="""
            Function/function name to generate dataloader for calibration,
            required'
        """,
    ),
    "weight_type": PassConfigParam(
        type_=str,
        default_value="QInt8",
        searchable_values=Categorical(["QInt8"]),
        description="""
            Data type for quantizing weights which is used in vai_q_onnx quantization.
            'QInt8' for signed 8-bit integer,
        """,
    ),
    "input_nodes": PassConfigParam(
        type_=list,
        default_value=None,
        description="""
            Start node that needs quantization. If None, all quantizable.
        """,
    ),
    "output_nodes": PassConfigParam(
        type_=list,
        default_value=None,
        description="""
            End node that needs quantization. If None, all quantizable.
        """,
    ),
    "op_types_to_quantize": PassConfigParam(
        type_=list,
        default_value=None,
        description="""
            List of operator types to quantize. If None, all quantizable.
        """,
    ),
    "nodes_to_quantize": PassConfigParam(
        type_=list,
        default_value=None,
        description="""
            List of node names to quantize. If None, all quantizable.
        """,
    ),
    "nodes_to_exclude": PassConfigParam(
        type_=list,
        default_value=None,
        description="""
            List of node names to exclude from quantization. If None, all quantizable.
        """,
    ),
    "optimize_model": PassConfigParam(
        type_=bool,
        default_value=False,
        searchable_values=Boolean(),
        description="""
            Deprecating Soon in ONNX! Optimize model before quantization. NOT recommended, optimization will
            change the computation graph, making debugging of quantization loss difficult.
        """,
    ),
    # TODO: enable search if we support onnx external data format
    "use_external_data_format": PassConfigParam(
        type_=bool,
        default_value=False,
        description="""
            option used for large size (>2GB) model. Set to False by default.
        """,
    ),
    "quant_preprocess": PassConfigParam(
        type_=bool,
        default_value=True,
        searchable_values=Boolean(),
        description="""
            Shape inference and model optimization, in preparation for quantization.
            https://onnxruntime.ai/docs/performance/quantization.html#pre-processing
        """,
    ),
    "calibrate_method": PassConfigParam(
        type_=str,
        default_value="MinMSE",
        searchable_values=Categorical(["NonOverflow", "MinMSE"]),
        description="""
            Current calibration methods supported are NonOverflow and MinMSE,
            Please use NonOverflow or MinMSE as options.
        """,
    ),
    "quant_format": PassConfigParam(
        type_=str,
        default_value="QDQ",
        searchable_values=Categorical(["QDQ"]),
        description="""
            QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
        """,
    ),
    "activation_type": PassConfigParam(
        type_=str,
        default_value="QInt8",
        # the search space is conditional on quant_format and weight_type
        # the equivalent joint search space for (quant_format, weight_type, activation) is
        # {(QDQ, QInt8, QInt8), (QDQ, QUInt8, QUInt8), (QOperator, QUInt8, QUInt8)}
        searchable_values=Conditional(
            parents=("quant_format", "weight_type"),
            support={
                ("QDQ", "QInt8"): Categorical(["QInt8"]),
            },
        ),
        description="""
            Quantization data type of activation.
        """,
    ),
}

_exposed_extra_options_config = {
    "ActivationSymmetric": PassConfigParam(
        type_=bool, default_value=True, description="symmetrize calibration data for activations"
    ),
    "WeightSymmetric": PassConfigParam(
        type_=bool, default_value=True, description="symmetrize calibration data for weights"
    ),
    "AddQDQPairToWeight": PassConfigParam(
        type_=bool,
        default_value=True,
        description="remains floating-point weight and inserts both QuantizeLinear/DeQuantizeLinear nodes to weight",
    ),
}

_extra_options_config = {
    "extra_options": PassConfigParam(
        type_=dict,
        default_value=None,
        description=f"""
            Key value pair dictionary for `extra_options` in quantization. If an option is one of
            {list(_exposed_extra_options_config.keys())}, it will be overwritten by the corresponding config parameter
            value.
        """,
    ),
}


class VitisAIQuantization(Pass):
    """
    Quantize ONNX model with onnxruntime where we can search for
    best parameters for vai_q_onnx quantization at same time.
    """

    _requires_user_script = True

    def _initialize(self):
        super()._initialize()
        self.tmp_dir = tempfile.TemporaryDirectory()

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "quant_mode": PassConfigParam(
                type_=str,
                default_value="static",
                searchable_values=Categorical(["static"]),
                description="""
                    Onnx Quantization mode. ,
                    'static' for vitis ai quantization.
                """,
            )
        }

        # common quantization config
        config.update(deepcopy(vai_q_onnx_quantization_config))

        # exposed extra options config
        config.update(deepcopy(_exposed_extra_options_config))
        config.update(deepcopy(_extra_options_config))
        return config

    def validate_search_point(self, search_point: Dict[str, Any]) -> bool:
        config = self.config_at_search_point(search_point)
        if config["quant_mode"] == "static":
            if (
                config["weight_type"] == "QInt8"
                and config["activation_type"] == "QInt8"
                and config["quant_format"] == "QOperator"
            ):
                logger.info("QOperator is not supported for Vitis AI Quantization.")
                return False
            if config["weight_type"] != "QInt8" or config["activation_type"] != "QInt8":
                logger.info("Weight type and activation type must be the QInt8.")
                return False
        return True

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        # start with a copy of the config
        run_config = deepcopy(config)

        output_model_path = ONNXModel.resolve_path(output_model_path)

        # extra config
        extra_options = deepcopy(config["extra_options"]) if config["extra_options"] else {}
        # keys in extra_options that are already exposed
        intersection = set(extra_options.keys()).intersection(set(_exposed_extra_options_config.keys()))
        if intersection:
            message = (
                f"Extra config keys {intersection} are already exposed in the pass config. They will be overwritten by"
                " the corresponding pass config parameter values."
            )
            logger.warning(message)
        for key in _exposed_extra_options_config:
            extra_options[key] = run_config[key]
            del run_config[key]

        # preprocess the model
        preprocessed_temp_model_path = Path(self.tmp_dir.name) / f"{Path(model.model_path).stem}_preprocessed.onnx"
        if run_config["quant_preprocess"]:
            if not preprocessed_temp_model_path.exists():
                # overwrite the model path with the preprocessed model path
                logger.info("Preprocessing model for quantization")
                model = self._quant_preprocess(model, preprocessed_temp_model_path)
            else:
                logger.info("Already processed model for quantization, skipping preprocessing")
                model = ONNXModel(preprocessed_temp_model_path)

        # keys not needed for quantization
        to_delete = [
            "quant_mode",
            "script_dir",
            "user_script",
            "quant_preprocess",
            "data_dir",
            "batch_size",
            "dataloader_func",
        ]

        # update string values to enum values
        run_config.update(
            {
                "calibrate_method": PowerOfTwoMethod[run_config["calibrate_method"]],
                "quant_format": QuantFormat[run_config["quant_format"]],
                "activation_type": QuantType[run_config["activation_type"]],
                "weight_type": QuantType[run_config["weight_type"]],
                "extra_options": extra_options,
            }
        )

        # remove keys not needed for quantization
        for key in to_delete:
            if key in run_config:
                del run_config[key]

        # get the dataloader
        if config["dataloader_func"]:
            dataloader = self._user_module_loader.call_object(
                config["dataloader_func"],
                get_local_path(config["data_dir"]),
                config["batch_size"],
            )
        elif self._data_config:
            dataloader = self._data_config.to_data_container().create_calibration_dataloader()
        quantize_static(
            model_input=model.model_path,
            model_output=output_model_path,
            calibration_data_reader=dataloader,
            **run_config,
        )

        return ONNXModel(output_model_path)

    def _quant_preprocess(self, model: ONNXModel, output_model_path: str) -> ONNXModel:
        try:
            quant_pre_process(input_model_path=model.model_path, output_model_path=output_model_path, auto_merge=True)
        except Exception as e:
            logger.warning(f"failed to run quantization preprocessing with error of {e}")
            copyfile(model.model_path, output_model_path)

        return ONNXModel(output_model_path)
