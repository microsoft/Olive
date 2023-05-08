# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, Union

import onnx

from olive.common.utils import hash_string
from olive.data.config import DataConfig
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
from olive.strategy.search_parameter import Boolean, Categorical, Conditional, ConditionalDefault

logger = logging.getLogger(__name__)

# common config for both static and dynamic quantization
_onnx_quantization_config = {
    "weight_type": PassConfigParam(
        type_=str,
        default_value="QInt8",
        searchable_values=Categorical(["QInt8", "QUInt8"]),
        description="""
            Data type for quantizing weights which is used both in dynamic
            and static quantization. 'QInt8' for signed 8-bit integer,
            'QUInt8' for unsigned 8-bit integer.
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
            Tips: When to use reduce_range and per-channel quantization:
            https://onnxruntime.ai/docs/performance/quantization.html#when-to-use-reduce-range-and-per-channel-quantization
        """,
    ),
    "reduce_range": PassConfigParam(
        type_=bool,
        default_value=False,
        searchable_values=Boolean(),
        description="""
            Quantize weights with 7-bits. It may improve the accuracy for
            some models running on non-VNNI machine, especially for per-channel mode.
            Tips: When to use reduce_range and per-channel quantization:
            https://onnxruntime.ai/docs/performance/quantization.html#when-to-use-reduce-range-and-per-channel-quantization
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
    "quant_preprocess": PassConfigParam(
        type_=bool,
        default_value=True,
        searchable_values=Boolean(),
        description="""
            Shape inference and model optimization, in preparation for quantization.
            https://onnxruntime.ai/docs/performance/quantization.html#pre-processing
        """,
    ),
}

_exposed_extra_options_config = {
    "extra.Sigmoid.nnapi": PassConfigParam(type_=bool, default_value=False, description=""),
    "ActivationSymmetric": PassConfigParam(
        type_=bool, default_value=False, description="symmetrize calibration data for activations"
    ),
    "WeightSymmetric": PassConfigParam(
        type_=bool, default_value=True, description="symmetrize calibration data for weights"
    ),
    "EnableSubgraph": PassConfigParam(
        type_=bool,
        default_value=False,
        description="If enabled, subgraph will be quantized. Dynamic mode currently is supported.",
    ),
    "ForceQuantizeNoInputCheck": PassConfigParam(
        type_=bool,
        default_value=False,
        description="""
            By default, some latent operators like maxpool, transpose, do not quantize if their input is not
            quantized already. Setting to True to force such operator always quantize input and so generate
            quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.
        """,
    ),
    "MatMulConstBOnly": PassConfigParam(
        type_=bool,
        default_value=ConditionalDefault(parents=("quant_mode",), support={("dynamic",): True, ("static",): False}),
        description="If enabled, only MatMul with const B will be quantized.",
    ),
}

_extra_options_config = {
    "extra_options": PassConfigParam(
        type_=dict,
        default_value=None,
        description=f"""
            Key value pair dictionary for `extra_options` in quantization. Please refer to
            https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py
            for details about the supported options. If an option is one of
            {list(_exposed_extra_options_config.keys())}, it will be overwritten by the corresponding config parameter
            value.
        """,
    ),
}

# static quantization specific config
_static_dataloader_config = {
    "data_dir": PassConfigParam(
        type_=Union[Path, str],
        is_path=True,
        description="""
            Path to the directory containing the dataset.
            For local data, it is required if quant_mode is 'static'.
        """,
    ),
    "batch_size": PassConfigParam(
        type_=int,
        default_value=1,
        description="""
            Batch size for calibration, required if quant_mode is 'static'.
        """,
    ),
    # TODO: remove this option once we have a data config ready
    "dataloader_func": PassConfigParam(
        type_=Union[Callable, str],
        required=False,
        is_object=True,
        description="""
            Function/function name to generate dataloader for calibration,
            required if quant_mode is 'static'
        """,
    ),
    "data_config": PassConfigParam(
        type_=Union[DataConfig, str],
        required=False,
        description="""
            Data config for calibration, required if quant_mode is 'static'.
            If not provided, a default DataConfig will be used.
        """,
    ),
}

_static_optional_config = {
    "calibrate_method": PassConfigParam(
        type_=str,
        default_value="MinMax",
        searchable_values=Categorical(["MinMax", "Entropy", "Percentile"]),
        description="""
            Current calibration methods supported are MinMax and Entropy,
            Please use CalibrationMethod.MinMax or CalibrationMethod.Entropy as options.
        """,
    ),
    "quant_format": PassConfigParam(
        type_=str,
        default_value="QDQ",
        searchable_values=Categorical(["QOperator", "QDQ"]),
        description="""
            QOperator format quantizes the model with quantized operators directly.
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
                ("QDQ", "QUInt8"): Categorical(["QUInt8"]),
                ("QOperator", "QUInt8"): Categorical(["QUInt8"]),
                # invalid choice for QOperator, QInt8
                ("QOperator", "QInt8"): Conditional.get_invalid_choice(),
            },
        ),
        description="""
            Quantization data type of activation. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        """,
    ),
}


class OnnxQuantization(Pass):
    """
    Quantize ONNX model with onnxruntime where we can search for
    best parameters for static/dynamic quantization at same time.
    """

    _requires_user_script = True

    def _initialize(self):
        super()._initialize()
        self.tmp_dir = tempfile.TemporaryDirectory()

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "quant_mode": PassConfigParam(
                type_=str,
                default_value="static",
                searchable_values=Categorical(["dynamic", "static"]),
                description="""
                    Onnx Quantization mode. 'dynamic' for dynamic quantization,
                    'static' for static quantization.
                """,
            )
        }

        # common quantization config
        config.update(deepcopy(_onnx_quantization_config))

        # static quantization config
        config.update(deepcopy(_static_dataloader_config))
        static_optional_config = deepcopy(_static_optional_config)
        for _, value in static_optional_config.items():
            # default value is conditional on quant_mode
            # if quant_mode is static, use the default value in static_optional_config
            # if quant_mode is dynamic, set default value as ignored. dynamic quantization doesn't use this parameter
            value.default_value = ConditionalDefault(
                parents=("quant_mode",),
                support={("static",): value.default_value, ("dynamic",): ConditionalDefault.get_ignored_choice()},
            )
            if isinstance(value.searchable_values, Categorical):
                # ignore the parameter if quant_mode is dynamic
                # if quant_mode is static, use the searchable_values in static_optional_config by making it conditional
                value.searchable_values = Conditional(
                    parents=("quant_mode",),
                    support={("static",): value.searchable_values},
                    default=Conditional.get_ignored_choice(),
                )
            elif isinstance(value.searchable_values, Conditional):
                # ignore the parameter if quant_mode is dynamic
                # if quant_mode is static, use the searchable_values in static_optional_config by expanding the parents
                value.searchable_values = Conditional(
                    parents=("quant_mode",) + value.searchable_values.parents,
                    support={
                        ("static",) + key: value.searchable_values.support[key]
                        for key in value.searchable_values.support
                    },
                    default=Conditional.get_ignored_choice(),
                )
        config.update(static_optional_config)

        # exposed extra options config
        config.update(deepcopy(_exposed_extra_options_config))
        config.update(deepcopy(_extra_options_config))

        # external data config
        config.update(get_external_data_config())
        return config

    def validate_search_point(self, search_point: Dict[str, Any]) -> bool:
        config = self.config_at_search_point(search_point)
        if config["quant_mode"] == "static":
            if (
                config["weight_type"] == "QInt8"
                and config["activation_type"] == "QInt8"
                and config["quant_format"] == "QOperator"
            ):
                logger.info("QOperator is not supported for QInt8 activation and weight.")
                return False
            if config["weight_type"] != config["activation_type"]:
                logger.info("Weight type and activation type must be the same.")
                return False
            if config["EnableSubgraph"] is True:
                logger.info("EnabaleSubgraph is not supported for static quantization.")
                return False
        return True

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
        from onnxruntime.quantization.calibrate import CalibrationMethod

        # start with a copy of the config
        run_config = deepcopy(config)
        is_static = run_config["quant_mode"] == "static"

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
            Path(self.tmp_dir.name) / f"{hash_string(str(Path(model.model_path).resolve()))}_preprocessed.onnx"
        )
        if run_config["quant_preprocess"]:
            if not preprocessed_temp_model_path.exists():
                # overwrite the model path with the preprocessed model path
                logger.info("Preprocessing model for quantization")
                model = self._quant_preprocess(model, preprocessed_temp_model_path)
            else:
                logger.info("Already processed model for quantization, skipping preprocessing")
                model = ONNXModel(preprocessed_temp_model_path, model.name)

        # keys not needed for quantization
        to_delete = ["quant_mode", "script_dir", "user_script", "quant_preprocess"]
        to_delete += list(get_external_data_config().keys())

        # update string values to enum values
        if is_static:
            to_delete += list(_static_dataloader_config.keys())
            run_config.update(
                {
                    "calibrate_method": CalibrationMethod[run_config["calibrate_method"]],
                    "quant_format": QuantFormat[run_config["quant_format"]],
                    "activation_type": QuantType[run_config["activation_type"]],
                    "weight_type": QuantType[run_config["weight_type"]],
                    "extra_options": extra_options,
                }
            )
        else:
            to_delete += list(_static_dataloader_config.keys())
            to_delete += list(_static_optional_config.keys())
            run_config.update(
                {
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
        # TODO: don't default to use_external_data_format=True if the loading and saving model makes
        # the pass inefficient
        tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")
        tmp_dir_path = Path(tmp_dir.name)
        tmp_model_path = str(tmp_dir_path / Path(output_model_path).name)

        if is_static:
            # get the dataloader
            # TODO: only use data config
            if self._user_module_loader.user_module:
                dataloader = self._user_module_loader.call_object(
                    self._fixed_params["dataloader_func"],
                    self._fixed_params["data_dir"],
                    self._fixed_params["batch_size"],
                )
            elif self._fixed_params["data_config"]:
                dc_cls = DataConfig(**self._fixed_params["data_config"])
                dataloader = dc_cls.to_data_container().create_calibration_dataloader()
            quantize_static(
                model_input=model.model_path,
                model_output=tmp_model_path,
                calibration_data_reader=dataloader,
                use_external_data_format=True,
                **run_config,
            )
        else:
            quantize_dynamic(
                model_input=model.model_path, model_output=tmp_model_path, use_external_data_format=True, **run_config
            )

        # load the model
        onnx_model = onnx.load(tmp_model_path)
        # the model is loaded into memory, so it's safe to delete previously exported files
        tmp_dir.cleanup()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config, model.name)

    def _quant_preprocess(self, model: ONNXModel, output_model_path: str) -> ONNXModel:
        from onnxruntime.quantization.preprocess import quant_pre_process

        try:
            quant_pre_process(input_model_path=model.model_path, output_model_path=output_model_path, auto_merge=True)
        except Exception as e:
            logger.warning(f"failed to run quantization preprocessing with error of {e}")
            copyfile(model.model_path, output_model_path)

        return ONNXModel(output_model_path, model.name)


class OnnxDynamicQuantization(OnnxQuantization):
    """ONNX Dynamic Quantization Pass"""

    _requires_user_script = False

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "quant_mode": PassConfigParam(type_=str, default_value="dynamic", description="dynamic quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_onnx_quantization_config))
        # exposed extra options config
        config.update(deepcopy(_exposed_extra_options_config))
        config.update(deepcopy(_extra_options_config))
        # external data config
        config.update(get_external_data_config())
        return config


class OnnxStaticQuantization(OnnxQuantization):
    """ONNX Static Quantization Pass"""

    _requires_user_script = True

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "quant_mode": PassConfigParam(type_=str, default_value="static", description="static quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_onnx_quantization_config))
        # static quantization specific config
        config.update(deepcopy(_static_dataloader_config))
        config.update(deepcopy(_static_optional_config))
        # exposed extra options config
        config.update(deepcopy(_exposed_extra_options_config))
        config.update(deepcopy(_extra_options_config))
        # external data config
        config.update(get_external_data_config())
        return config


_inc_quantization_config = {
    "device": PassConfigParam(
        type_=str,
        default_value="cpu",
        description="""
            Intel® Neural Compressor quantization device. Support 'cpu' and 'gpu'.
        """,
    ),
    "backend": PassConfigParam(
        type_=str,
        default_value="default",
        description="""
            Backend for model execution. Support 'default', 'onnxrt_trt_ep', 'onnxrt_cuda_ep'
        """,
    ),
    "domain": PassConfigParam(
        type_=str,
        default_value="auto",
        description="""
            Model domain. Support 'auto', 'cv', 'object_detection', 'nlp' and 'recommendation_system'.
            Intel® Neural Compressor Adaptor will use specific quantization settings for different domains
            automatically, and explicitly specified quantization settings will override the automatic setting.
            If users set domain as auto, automatic detection for domain will be executed.
        """,
    ),
    "recipes": PassConfigParam(
        type_=dict,
        default_value={},
        description="""
            Recipes for Intel® Neural Compressor quantiztaion, support list is as below.
                'smooth_quant': whether do smooth quant
                'smooth_quant_args': parameters for smooth_quant
                'fast_bias_correction': whether do fast bias correction
                'weight_correction': whether do weight correction
                'gemm_to_matmul': whether convert gemm to matmul and add, only valid for onnx models
                'graph_optimization_level': support 'DISABLE_ALL', 'ENABLE_BASIC', 'ENABLE_EXTENDED', 'ENABLE_ALL'
                                        only valid for onnx models
                'first_conv_or_matmul_quantization': whether quantize the first conv or matmul
                'last_conv_or_matmul_quantization': whether quantize the last conv or matmul
                'pre_post_process_quantization': whether quantize the ops in preprocess and postprocess
                'add_qdq_pair_to_weight': whether add QDQ pair for weights, only vaild for onnxrt_trt_ep
                'optypes_to_exclude_output_quant': don't quantize output of specified optypes
                'dedicated_qdq_pair': whether dedicate QDQ pair, only vaild for onnxrt_trt_ep
        """,
    ),
    "reduce_range": PassConfigParam(
        type_=bool,
        default_value=False,
        searchable_values=Boolean(),
        description="""
            Whether use 7 bit to quantization.
        """,
    ),
    "quant_level": PassConfigParam(
        type_=str,
        default_value="auto",
        description="""
            Intel® Neural Compressor allows users to choose different tuning processes by specifying
            the quantization level (quant_level). Currently 3 quant_levels are supported.
            0 is conservative strategy, 1 is basic or user-specified strategy,
            auto (default) is the combination of 0 and 1.
            Please refer to
            https://github.com/intel/neural-compressor/blob/master/docs/source/tuning_strategies.md#tuning-process
            https://github.com/intel/neural-compressor/blob/master/docs/source/tuning_strategies.md#tuning-algorithms
            for more details
        """,
    ),
    "excluded_precisions": PassConfigParam(
        type_=list,
        default_value=[],
        description="""
            Precisions to be excluded, Default value is empty list.
            Intel® Neural Compressor enable the mixed precision with
            fp32 + bf16(only when device is 'gpu' and backend is 'onnxrt_cuda_ep') + int8 by default.
            If you want to disable bf16 data type, you can specify excluded_precisions = ['bf16'].
        """,
    ),
    "use_distributed_tuning": PassConfigParam(
        type_=bool,
        default_value=False,
        description="""
            Intel® Neural Compressor provides distributed tuning to speed up the tuning
            process by leveraging the multi-node cluster. Prerequisites: A working MPI
            implementation and installed mpi4py.
        """,
    ),
}

_inc_static_optional_config = {
    "quant_format": PassConfigParam(
        type_=str,
        default_value="QOperator",
        searchable_values=Categorical(["QOperator", "QDQ"]),
        description="""
            Quantization format. Support 'QDQ' and 'QOperator'.
        """,
    ),
    "calibration_sampling_size": PassConfigParam(
        type_=Union[list, int],
        default_value=[100],
        description="""
            Number of calibration sample.
        """,
    ),
}


class IncQuantization(Pass):
    """
    Quantize ONNX model with Intel® Neural Compressor.
    """

    _requires_user_script = True

    def _initialize(self):
        super()._initialize()

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "approach": PassConfigParam(
                type_=str,
                default_value="static",
                searchable_values=Categorical(["dynamic", "static"]),
                description="""
                Intel® Neural Compressor Quantization mode. 'dynamic' for dynamic quantization,
                'static' for static quantization.
            """,
            )
        }

        # common quantization config
        config.update(deepcopy(_inc_quantization_config))

        # static quantization config
        config.update(deepcopy(_static_dataloader_config))
        inc_static_optional_config = deepcopy(_inc_static_optional_config)
        for _, value in inc_static_optional_config.items():
            # default value of quant_format is conditional on approach
            if isinstance(value.searchable_values, Categorical):
                # ignore the parameter quant_format if approach is dynamic, if approach is static,
                # use the searchable_values in inc_static_optional_config by making it conditional
                value.searchable_values = Conditional(
                    parents=("approach",),
                    support={("static",): value.searchable_values},
                    default=Categorical(["default"]),
                )
            elif isinstance(value.searchable_values, Conditional):
                # ignore the parameter quant_format if approach is dynamic, if approach is static,
                # use the searchable_values in inc_static_optional_config by expanding the parents
                value.searchable_values = Conditional(
                    parents=("approach",) + value.searchable_values.parents,
                    support={
                        ("static",) + key: value.searchable_values.support[key]
                        for key in value.searchable_values.support
                    },
                    default=Categorical(["default"]),
                )
        config.update(inc_static_optional_config)
        return config

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:

        try:
            from neural_compressor import PostTrainingQuantConfig, quantization
        except ImportError:
            raise ImportError("Please install neural-compressor to use Intel® Neural Compressor quantization")

        # start with a copy of the config
        run_config = deepcopy(config)
        is_static = run_config["approach"] == "static"

        # add onnx extension if not present
        if Path(output_model_path).suffix != ".onnx":
            output_model_path += ".onnx"

        # keys not needed for quantization
        to_delete = ["script_dir", "user_script", "data_dir", "batch_size", "dataloader_func", "data_config"]
        for key in to_delete:
            if key in run_config:
                del run_config[key]

        config = PostTrainingQuantConfig(**run_config)
        inc_calib_dataloader = (
            self._user_module_loader.call_object(
                self._fixed_params["dataloader_func"], self._fixed_params["data_dir"], self._fixed_params["batch_size"]
            )
            if is_static
            else None
        )
        inc_model = model.load_model()
        q_model = quantization.fit(inc_model, config, calib_dataloader=inc_calib_dataloader)
        q_model.save(output_model_path)

        return ONNXModel(output_model_path, model.name)


class IncDynamicQuantization(IncQuantization):
    """Intel® Neural Compressor Dynamic Quantization Pass"""

    _requires_user_script = False

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        config = {
            "approach": PassConfigParam(type_=str, default_value="dynamic", description="dynamic quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_inc_quantization_config))
        return config


class IncStaticQuantization(IncQuantization):
    """Intel® Neural Compressor Static Quantization Pass"""

    _requires_user_script = True

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        config = {
            "approach": PassConfigParam(type_=str, default_value="static", description="static quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_inc_quantization_config))
        # static quantization specific config
        config.update(deepcopy(_static_dataloader_config))
        config.update(deepcopy(_inc_static_optional_config))
        return config
