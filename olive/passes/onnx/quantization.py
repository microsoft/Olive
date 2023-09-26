# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Union

import onnx
from packaging import version

from olive.cache import get_local_path_from_root
from olive.common.config_utils import validate_config
from olive.common.utils import hash_string
from olive.data.config import DataConfig
from olive.exception import OlivePassError
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_file, model_proto_to_olive_model
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, LocalFile
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
        type_=OLIVE_RESOURCE_ANNOTATIONS,
        category=ParamCategory.DATA,
        description="""
            Path to the directory containing the dataset.
            For local data, it is required if quant_mode is 'static' and dataloader_func is provided.
        """,
    ),
    "batch_size": PassConfigParam(
        type_=int,
        default_value=1,
        description="""
            Batch size for calibration, only used if dataloader_func is provided.
        """,
    ),
    # TODO(trajep): remove this option once we have a data config ready
    "dataloader_func": PassConfigParam(
        type_=Union[Callable, str],
        category=ParamCategory.OBJECT,
        description="""
            Function/function name to generate dataloader for calibration,
            required if quant_mode is 'static' and data_config is None.
        """,
    ),
    "data_config": PassConfigParam(
        type_=Union[DataConfig, Dict],
        description="""
            Data config for calibration, required if quant_mode is 'static' and
            dataloader_func is None.
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
            Percentile is not supported for onnxruntime==1.16.0, please avoid to set/search it.
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
    """Quantize ONNX model with static/dynamic quantization techniques."""

    _requires_user_script = True

    def _initialize(self):
        super()._initialize()
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
        for value in static_optional_config.values():
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
                    parents=("quant_mode", *value.searchable_values.parents),
                    support={
                        ("static", *key): value.searchable_values.support[key]
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

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        config = search_point or {}
        if with_fixed_value:
            config = self.config_at_search_point(search_point)
        if config["quant_mode"] == "static":
            if (
                config["weight_type"] == "QInt8"
                and config["activation_type"] == "QInt8"
                and config["quant_format"] == "QOperator"
            ):
                logger.info("S8S8 with QOperator will be slow on x86-64 CPUs and should be avoided in general")
                return False
            if config["EnableSubgraph"] is True:
                logger.info("EnableSubgraph is not supported for static quantization.")
                return False
        return True

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        from onnxruntime import __version__ as OrtVersion
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
        from onnxruntime.quantization.calibrate import CalibrationMethod

        # start with a copy of the config
        run_config = deepcopy(config)
        is_static = run_config["quant_mode"] == "static"
        if is_static:
            assert (
                config["dataloader_func"] or config["data_config"]
            ), "dataloader_func or data_config is required for static quantization."

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
        to_delete = ["quant_mode", "script_dir", "user_script", "quant_preprocess", "data_config"]
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

        # for ORT version < 1.16.0, set optimize_model to False
        # always set it to False since it is not recommended and is removed in ORT 1.16.0
        # user needs to call pre-process to optimize the model, we already have pre-process option
        if version.parse(OrtVersion) < version.parse("1.16.0"):
            run_config["optimize_model"] = False

        # to be safe, run the quantizer with use_external_data_format set to `True` and
        # `model_output` to a temporary directory
        # reload the model and save to output_model_path using the external data config
        # TODO(jambayk): don't default to use_external_data_format=True if the loading and saving model makes
        # the pass inefficient
        new_tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")
        tmp_model_path = str(Path(new_tmp_dir.name) / Path(output_model_path).name)

        if is_static:
            # get the dataloader
            # TODO(trajep): only use data config
            if config["dataloader_func"]:
                data_dir = get_local_path_from_root(data_root, config["data_dir"])
                dataloader = self._user_module_loader.call_object(
                    config["dataloader_func"],
                    data_dir,
                    config["batch_size"],
                )
            elif config["data_config"]:
                data_config = validate_config(config["data_config"], DataConfig)
                dataloader = data_config.to_data_container().create_calibration_dataloader(data_root)
            try:
                quantize_static(
                    model_input=model.model_path,
                    model_output=tmp_model_path,
                    calibration_data_reader=dataloader,
                    use_external_data_format=True,
                    **run_config,
                )
            except (AttributeError, ValueError) as e:
                raise OlivePassError("quantize_static failed.") from e
        else:
            try:
                quantize_dynamic(
                    model_input=model.model_path,
                    model_output=tmp_model_path,
                    use_external_data_format=True,
                    **run_config,
                )
            except (AttributeError, ValueError) as e:
                raise OlivePassError("quantize_dynamic failed.") from e

        # load the model
        onnx_model = onnx.load(tmp_model_path)
        # the model is loaded into memory, so it's safe to delete previously exported files
        # NOTE: Don't cleanup self.tmp_dir to avoid preprocessing the same model again during
        # recurrent passes of the search.
        new_tmp_dir.cleanup()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    def _quant_preprocess(self, model: ONNXModel, output_model_path: Union[str, Path]) -> ONNXModel:
        from olive.passes.onnx.quant_pre_process import quant_pre_process

        try:
            # TODO(myguo): use ORT version once the Windows issue is fixed
            quant_pre_process(
                input_model_path=model.model_path,
                output_model_path=str(output_model_path),
                auto_merge=True,
                save_as_external_data=True,
            )
        except Exception as e:
            # TODO(jambayk): try with `skip_optimization = True`
            # quantization preprocessing will fail if the model is too large and `skip_optimization = False`
            # there are some problems with the path to where the external data is saved
            # need to find out why before enabling this

            logger.warning(
                f"Failed to run quantization preprocessing with error of {e}. Using original model.", exc_info=True
            )
            # save original model to output path
            onnx_model = onnx.load(model.model_path)
            model_proto_to_file(
                onnx_model,
                output_model_path,
                save_as_external_data=True,  # always save as external data to avoid failures due to large models
            )

        # since this is only used internally, we will just treat it as a model file
        return ONNXModel(LocalFile({"path": output_model_path}))


class OnnxDynamicQuantization(OnnxQuantization):
    """ONNX Dynamic Quantization Pass."""

    _requires_user_script = False

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
    """ONNX Static Quantization Pass."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
