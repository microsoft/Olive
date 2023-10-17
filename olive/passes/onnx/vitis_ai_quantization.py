#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Union

import onnx

from olive.cache import get_local_path_from_root
from olive.common.utils import hash_string
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_file, model_proto_to_olive_model
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, LocalFile
from olive.strategy.search_parameter import Boolean, Categorical, Conditional

logger = logging.getLogger(__name__)

# common config for Vitis-AI quantization
vai_q_onnx_quantization_config = {
    "data_dir": PassConfigParam(
        type_=OLIVE_RESOURCE_ANNOTATIONS,
        category=ParamCategory.DATA,
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
        category=ParamCategory.OBJECT,
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
    "per_channel": PassConfigParam(
        type_=bool,
        default_value=False,
        searchable_values=Boolean(),
        description="""
            Quantize weights per channel.
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
    # TODO(xiaosheng): enable search if we support onnx external data format
    "use_external_data_format": PassConfigParam(
        type_=bool,
        default_value=True,
        description="""
            option used for large size (>2GB) model. Set to True by default.
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
        searchable_values=Categorical(["QDQ", "QOperator"]),
        description="""
            QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
        """,
    ),
    "need_layer_fusing": PassConfigParam(
        type_=bool,
        default_value=False,
        searchable_values=Boolean(),
        description="""
            Perform layer fusion for conv-relu type operations
        """,
    ),
    "activation_type": PassConfigParam(
        type_=str,
        default_value="QUInt8",
        # the search space is conditional on quant_format and weight_type
        # the equivalent joint search space for (quant_format, weight_type, activation) is
        # {(QDQ, QInt8, QInt8), (QDQ, QUInt8, QUInt8), (QOperator, QUInt8, QUInt8)}
        searchable_values=Conditional(
            parents=("quant_format", "weight_type"),
            support={
                ("QDQ", "QInt8"): Categorical(["QInt8"]),
                ("QDQ", "QUInt8"): Categorical(["QUInt8"]),
                ("QOperator", "QUInt8"): Categorical(["QUInt8"]),
                # invalid choice for QOperator, QInt8
                ("QOperator", "QInt8"): Conditional.get_invalid_choice(),
            },
        ),
        description="""
            Quantization data type of activation.
        """,
    ),
    "enable_dpu": PassConfigParam(
        type_=bool,
        default_value=False,
        searchable_values=Boolean(),
        description="""
            Use QDQ format optimized specifically for DPU.
        """,
    ),
}

_exposed_extra_options_config = {
    "ActivationSymmetric": PassConfigParam(
        type_=bool, default_value=False, description="symmetrize calibration data for activations"
    ),
    "WeightSymmetric": PassConfigParam(
        type_=bool, default_value=True, description="symmetrize calibration data for weights"
    ),
    "AddQDQPairToWeight": PassConfigParam(
        type_=bool,
        default_value=False,
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
    """Quantize ONNX model with onnxruntime.

    We can search for best parameters for vai_q_onnx quantization at same time.
    """

    _requires_user_script = True

    def _initialize(self):
        super()._initialize()
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_vaiq_tmp")

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Override this method to return False by using the accelerator spec information."""
        return False

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "quant_mode": PassConfigParam(
                type_=str,
                default_value="static",
                searchable_values=Categorical(["static"]),
                description="""
                    Onnx Quantization mode.
                    'static' for vitis ai quantization.
                """,
            )
        }

        # common quantization config
        config.update(deepcopy(vai_q_onnx_quantization_config))

        # exposed extra options config
        config.update(deepcopy(_exposed_extra_options_config))
        config.update(deepcopy(_extra_options_config))

        # external data config
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

        from olive.passes.onnx.vitis_ai import quantize_static
        from olive.passes.onnx.vitis_ai.quant_utils import PowerOfTwoMethod

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
        # we hash the entire path of the input model to ensure we are not accidentally using a preprocessed model
        # from a different model
        preprocessed_temp_model_path = (
            Path(self.tmp_dir.name) / f"{hash_string(str(Path(model.model_path).resolve()))}" / "preprocessed.onnx"
        )
        preprocessed_temp_model_path.parent.mkdir(exist_ok=True, parents=True)
        if run_config["quant_preprocess"]:
            if not preprocessed_temp_model_path.exists():
                # overwrite the model path with the preprocessed model path
                logger.info("Preprocessing model for quantization")
                model = self._quant_preprocess(model, preprocessed_temp_model_path)
            else:
                logger.info("Already processed model for quantization, skipping preprocessing")
                model = ONNXModel(LocalFile({"path": preprocessed_temp_model_path}))

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
        to_delete += list(get_external_data_config().keys())

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

        # to be safe, run the quantizer with use_external_data_format set to `True` and
        # `model_output` to a temporary directory
        # reload the model and save to output_model_path using the external data config
        # TODO(XiaoSheng): don't default to use_external_data_format=True if the loading and saving model makes
        # the pass inefficient
        tmp_dir = tempfile.TemporaryDirectory(prefix="olive_vaiq_tmp")
        tmp_dir_path = Path(tmp_dir.name)
        tmp_model_path = str(tmp_dir_path / Path(output_model_path).name)

        # get the dataloader
        # TODO(XiaoSheng): only use data config
        if config["dataloader_func"]:
            data_dir = get_local_path_from_root(data_root, config["data_dir"])
            dataloader = self._user_module_loader.call_object(
                config["dataloader_func"],
                data_dir,
                config["batch_size"],
            )
        elif self._data_config:
            dataloader = self._data_config.to_data_container().create_calibration_dataloader(data_root)

        execution_provider = self.accelerator_spec.execution_provider

        quantize_static(
            model_input=model.model_path,
            model_output=tmp_model_path,
            calibration_data_reader=dataloader,
            execution_providers=[execution_provider],
            **run_config,  # use_external_data_format has been set to `True` by default in run_config
        )
        # load the model
        onnx_model = onnx.load(tmp_model_path)
        # the model is loaded into memory, so it's safe to delete previously exported files
        tmp_dir.cleanup()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    def _quant_preprocess(self, model: ONNXModel, output_model_path: str) -> ONNXModel:
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

            logger.warning(f"Failed to run quantization preprocessing with error of {e}. Using original model.")
            # save original model to output path
            onnx_model = onnx.load(model.model_path)
            model_proto_to_file(
                onnx_model,
                output_model_path,
                save_as_external_data=True,  # always save as external data to avoid failures due to large models
            )

        # since this is only used internally, we will just treat it as a model file
        return ONNXModel(LocalFile({"path": output_model_path}))
