# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Union

import onnx
from packaging import version

from olive.cache import get_local_path_from_root
from olive.common.config_utils import validate_config
from olive.common.pydantic_v1 import validator
from olive.common.utils import hash_string
from olive.data.config import DataConfig
from olive.exception import OlivePassError
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_file, model_proto_to_olive_model
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, LocalFile
from olive.strategy.search_parameter import Boolean, Categorical, Conditional, ConditionalDefault

logger = logging.getLogger(__name__)

# pylint: disable=consider-using-with

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
_dataloader_config = {
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
    "dataloader_func_kwargs": PassConfigParam(
        type_=Dict[str, Any],
        description="Keyword arguments for dataloader_func.",
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
    "prepare_qnn_config": PassConfigParam(
        type_=bool,
        default_value=False,
        description="""
            Whether to generate a suitable quantization config for the input model.
            Should be set to True if model is targeted for QNN EP.
        """,
    ),
}


def get_calibration_dataloader(data_root, user_module_loader, config):
    if config["dataloader_func"]:
        # TODO(trajep): replace legacy dataloader_func with data config
        data_dir = get_local_path_from_root(data_root, config["data_dir"])
        dataloader = user_module_loader.call_object(
            config["dataloader_func"],
            data_dir,
            config["batch_size"],
            **(config["dataloader_func_kwargs"] or {}),
        )
    elif config["data_config"]:
        data_config = validate_config(config["data_config"], DataConfig)
        dataloader = data_config.to_data_container().create_calibration_dataloader(data_root)
    return dataloader


class OnnxQuantization(Pass):
    """Quantize ONNX model with static/dynamic quantization techniques."""

    _requires_user_script = True

    def _initialize(self):
        super()._initialize()
        # pylint: disable=attribute-defined-outside-init
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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
        config.update(deepcopy(_dataloader_config))
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
                # S8S8 with QOperator will be slow on x86-64 CPUs and should be avoided in general.
                # https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection
                # But we still allow it for users to try at their own risk. Olive just warns this to users.
                logger.warning(
                    "S8S8 with QOperator will be slow on x86-64 CPUs and should be avoided in general, try QDQ instead."
                )
            if config["EnableSubgraph"] is True:
                logger.info("EnableSubgraph is not supported for static quantization.")
                return False
        return True

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
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
            # whether to prepare qnn config
            # we do the version check here and not in `validate_search_point` since search point validation
            # is done by the engine. Unless the host is local system, the ort version of the host is
            # not known by the engine when the search point is validated.
            if config["prepare_qnn_config"] and version.parse(OrtVersion) < version.parse("1.17.0"):
                raise OlivePassError("prepare_qnn_config is only supported by onnxruntime>=1.17.0")

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # extra config
        extra_options = deepcopy(config["extra_options"]) if config["extra_options"] else {}
        # keys in extra_options that are already exposed
        intersection = set(extra_options.keys()).intersection(set(_exposed_extra_options_config.keys()))
        if intersection:
            logger.warning(
                "Extra config keys %s are already exposed in the pass config. They will be overwritten by"
                " the corresponding pass config parameter values.",
                intersection,
            )
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
                model = ONNXModelHandler(LocalFile({"path": preprocessed_temp_model_path}))

        # keys not needed for quantization
        to_delete = [
            "quant_mode",
            "script_dir",
            "user_script",
            "quant_preprocess",
            "data_config",
            "prepare_qnn_config",
        ]
        to_delete += list(get_external_data_config().keys())

        # update string values to enum values
        if is_static:
            to_delete += list(_dataloader_config.keys())
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
            to_delete += list(_dataloader_config.keys())
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
            dataloader = get_calibration_dataloader(data_root, self._user_module_loader, config)
            if config["prepare_qnn_config"]:
                import inspect

                from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

                qnn_config = get_qnn_qdq_config(
                    model_input=model.model_path,
                    calibration_data_reader=dataloader,
                    calibrate_method=run_config["calibrate_method"],
                    activation_type=run_config["activation_type"],
                    weight_type=run_config["weight_type"],
                    per_channel=run_config["per_channel"],
                )
                # override the run_config with qnn_config
                # get all attributes of qnn_config
                run_config = {k: v for k, v in inspect.getmembers(qnn_config) if not k.startswith("_")}
                # remove the calibration_data_reader from run_config
                run_config.pop("calibration_data_reader", None)

            for key in ("calibration_data_reader", "use_external_data_format"):
                if key in run_config:
                    del run_config[key]
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

    def _quant_preprocess(self, model: ONNXModelHandler, output_model_path: Union[str, Path]) -> ONNXModelHandler:
        from onnxruntime.quantization.preprocess import quant_pre_process

        try:
            quant_pre_process(
                input_model_path=model.model_path,
                output_model_path=str(output_model_path),
                auto_merge=True,
                save_as_external_data=True,
                verbose=3,  # set verbose to 3 to get more information about the preprocessing
            )
        except Exception as e:
            # TODO(jambayk): try with `skip_optimization = True`
            # quantization preprocessing will fail if the model is too large and `skip_optimization = False`
            # there are some problems with the path to where the external data is saved
            # need to find out why before enabling this

            logger.warning(
                "Failed to run quantization preprocessing with error of %s. Using original model.", e, exc_info=True
            )
            # save original model to output path
            onnx_model = onnx.load(model.model_path)
            model_proto_to_file(
                onnx_model,
                output_model_path,
                save_as_external_data=True,  # always save as external data to avoid failures due to large models
            )

        # since this is only used internally, we will just treat it as a model file
        return ONNXModelHandler(LocalFile({"path": output_model_path}))


class OnnxDynamicQuantization(OnnxQuantization):
    """ONNX Dynamic Quantization Pass."""

    _requires_user_script = False

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        if accelerator_spec.execution_provider == "QNNExecutionProvider":
            raise ValueError("QNNExecutionProvider is not supported for dynamic quantization.")
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

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "quant_mode": PassConfigParam(type_=str, default_value="static", description="static quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_onnx_quantization_config))
        # static quantization specific config
        config.update(deepcopy(_dataloader_config))
        config.update(deepcopy(_static_optional_config))
        # exposed extra options config
        config.update(deepcopy(_exposed_extra_options_config))
        config.update(deepcopy(_extra_options_config))
        # external data config
        config.update(get_external_data_config())
        if accelerator_spec.execution_provider == "QNNExecutionProvider":
            config["quant_format"].searchable_values = Categorical(["QDQ"])
            # Recently Int16/Uint16 is added into onnx runtime quantization only in QDQ mode.
            # for QNN EP integration, we give this workaround to support Int16/Uint16 in QDQ mode.
            # TODO(jiapli): remove this workaround once figure out the Int16/UInt16 in latest quantization
            config["activation_type"].searchable_values = Categorical(["QInt8", "QUInt8", "QUInt16", "QInt16"])
            config["weight_type"].searchable_values = Categorical(["QInt8", "QUInt8", "QUInt16", "QInt16"])
            config["prepare_qnn_config"].default_value = True
            config["quant_preprocess"].default_value = False
        return config


class OnnxMatMul4Quantizer(Pass):
    _requires_user_script = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "block_size": PassConfigParam(
                type_=int,
                default_value=32,
                description="Block size for quantization. Default value is 32.",
            ),
            "is_symmetric": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Symmetric quantization. Default value is True.",
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of node names to exclude from quantization.",
            ),
            "accuracy_level": PassConfigParam(
                # TODO(trajep): to make it searchable
                type_=int,
                default_value=None,
                description="Available from onnxruntime>=1.17.0 "
                "The minimum accuracy level of input A, can be: 0(unset), 1(fp32), 2(fp16), 3(bf16), "
                "or 4(int8) (default unset when 0 or None). It is used to control how input A is quantized or downcast "
                "internally while doing computation, for example: 0 means input A will not be quantized "
                "or downcast while doing computation. 4 means input A can be quantized with the same "
                "block_size to int8 internally from type T1. "
                "Refer to the MatMulNBits contrib op's 'accuracy_level' attribute for details "
                "(https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits).",
            ),
            "algorithm": PassConfigParam(
                type_=str,
                default_value=None,
                description="If 'None', the Matmul node with fp32 const weight will be quantize to int4."
                "1. 'RTN' and 'GPTQ' are available from onnxruntime>=1.17.0 "
                "- For 4b quantize a model with RTN or GPTQ algorithm. Please refer to "
                "https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md "
                "for more details on weight only quantization using Intel® Neural Compressor. "
                "2. 'DEFAULT', 'HQQ' are available from onnxruntime>=1.18.0 "
                "- `DEFAULT` takes the same effect as `None`"
                "- For HQQ, please refer to onnxruntime for more details: "
                "https://github.com/microsoft/onnxruntime/blob/7e613ee821405b1192d0b71b9434a4f94643f1e4/onnxruntime/python/tools/quantization/matmul_4bits_quantizer.py#L102C1-L126C25",
            ),
            "weight_only_quant_configs": PassConfigParam(
                type_=dict,
                default_value=None,
                description="""Available from onnxruntime>=1.17.0, if None, the default behavior
                of given algorithm will be used.
                The config is binding to the algorithm with following map:
                1. "algorithm" is "DEFAULT", by default, the weight_only_quant_configs is:
                    "weight_only_quant_configs": {
                        "block_size": 128,
                        "is_symmetric": False,
                        "accuracy_level": None
                    }
                    https://github.com/microsoft/onnxruntime/blob/7e613ee821405b1192d0b71b9434a4f94643f1e4/onnxruntime/python/tools/quantization/matmul_4bits_quantizer.py#L129C1-L140C45
                2. "algorithm" is "HQQ", by default, the weight_only_quant_configs is:
                    "weight_only_quant_configs": {
                        "block_size": 128, // channel number in one block to execute a GPTQ quantization iteration.
                        "bits": 4, // how many bits to represent weight.
                        "axis": 1, // 0 or 1. which axis to quantize. https://arxiv.org/pdf/2309.15531.pdf
                    }
                    https://github.com/microsoft/onnxruntime/blob/7e613ee821405b1192d0b71b9434a4f94643f1e4/onnxruntime/python/tools/quantization/matmul_4bits_quantizer.py#L129C1-L140C45
                3. "algorithm" is "RTN", by default, the weight_only_quant_configs is:
                    "weight_only_quant_configs": {
                        "ratios": None, // type: dict, percentile of clip. Defaults to None.
                    }
                    https://github.com/microsoft/onnxruntime/blob/7e613ee821405b1192d0b71b9434a4f94643f1e4/onnxruntime/python/tools/quantization/matmul_4bits_quantizer.py#L42C1-L60C29
                4. "algorithm" is "GPTQ", by default, the weight_only_quant_configs is:
                    "weight_only_quant_configs": {
                        "percdamp": 0.01, // percent of the average Hessian diagonal to use for dampening.
                        "block_size": 128,
                        "actorder": False, // whether rearrange Hessian matrix considering the diag's value.
                        "mse": False, // whether get scale and zero point with mse error.
                        "perchannel": True, // whether quantize weight per-channel.
                    }
                    For GPTQ's "calibration_data_reader", you can provider a dataloader function or a
                    data config like what we do for onnx static quantization.
                    https://github.com/microsoft/onnxruntime/blob/7e613ee821405b1192d0b71b9434a4f94643f1e4/onnxruntime/python/tools/quantization/matmul_4bits_quantizer.py#L63C1-L99C37
                """,
            ),
        }
        config.update(get_external_data_config())
        # static_dataloder_config
        config.update(deepcopy(_dataloader_config))
        return config

    @classmethod
    def _validators(cls) -> Dict[str, Callable]:
        return {
            "validate_accuracy_level": validator("accuracy_level", allow_reuse=True)(_validate_accuracy_level),
            "validate_algorithm": validator("algorithm", allow_reuse=True)(_validate_algorithm),
            "validate_quant_config": validator("weight_only_quant_configs", allow_reuse=True)(
                _validate_weight_only_quant_config
            ),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime import __version__ as OrtVersion

        if version.parse(OrtVersion) < version.parse("1.16.2"):
            raise ValueError("MatMul4BitsQuantizer is only supported in onnxruntime >= 1.16.2")

        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        weight_only_quant_config_class = None
        weight_only_quant_config = None
        algo_config = deepcopy(config["weight_only_quant_configs"] or {})
        if version.parse(OrtVersion) >= version.parse("1.17.0"):
            from onnxruntime.quantization.matmul_4bits_quantizer import (
                GPTQWeightOnlyQuantConfig,
                RTNWeightOnlyQuantConfig,
            )

            if config["algorithm"] == "RTN":
                weight_only_quant_config_class = RTNWeightOnlyQuantConfig
            elif config["algorithm"] == "GPTQ":
                if "block_size" in algo_config and version.parse(OrtVersion) < version.parse("1.18.0"):
                    # ort 1.17.0+ uses blocksize instead of block_size :(
                    algo_config["blocksize"] = algo_config["block_size"]
                    algo_config.pop("block_size")
                dataloader = get_calibration_dataloader(data_root, self._user_module_loader, config)
                weight_only_quant_config_class = partial(GPTQWeightOnlyQuantConfig, calibration_data_reader=dataloader)

            if version.parse(OrtVersion) >= version.parse("1.18.0"):
                from onnxruntime.quantization.matmul_4bits_quantizer import (
                    DefaultWeightOnlyQuantConfig,
                    HQQWeightOnlyQuantConfig,
                )

                if config["algorithm"] == "DEFAULT":
                    weight_only_quant_config_class = DefaultWeightOnlyQuantConfig
                elif config["algorithm"] == "HQQ":
                    weight_only_quant_config_class = HQQWeightOnlyQuantConfig
            elif config["algorithm"] in ("HQQ", "DEFAULT"):
                raise ValueError("HQQ and DEFAULT algorithm are only supported in onnxruntime >= 1.18.0")

            if weight_only_quant_config_class:
                weight_only_quant_config = weight_only_quant_config_class(**algo_config)
            quant = MatMul4BitsQuantizer(
                model.load_model(),
                block_size=config["block_size"],
                is_symmetric=config["is_symmetric"],
                nodes_to_exclude=config["nodes_to_exclude"],
                accuracy_level=config["accuracy_level"],
                algo_config=weight_only_quant_config,
            )
        else:
            # TODO(trajep): remove this block once we migrate customer to onnxruntime>=1.17.0 all
            quant = MatMul4BitsQuantizer(
                model.load_model(),
                block_size=config["block_size"],
                is_symmetric=config["is_symmetric"],
                nodes_to_exclude=config["nodes_to_exclude"],
            )
        quant.process()
        # topologically sort the graph at the end since previous optimizations may have broken it
        quant.model.topological_sort()
        # quant.model._check_init is not needed since it's only meant for float8 quantization

        # save the model to the output path and return the model
        return model_proto_to_olive_model(quant.model.model, output_model_path, config)


def _validate_accuracy_level(v, values, field):
    if not v:
        return v

    if v not in (0, 1, 2, 3, 4):
        raise ValueError(f"OnnxMatMul4Quantizer {field.name} must be 0(unset), 1(fp32), 2(fp16), 3(bf16) or 4(int8)")

    return v


def _validate_algorithm(v, values, field):
    if not v:
        return v

    if v not in ("DEFAULT", "HQQ", "RTN", "GPTQ"):
        raise ValueError(f"OnnxMatMul4Quantizer {field.name} must be 'DEFAULT', 'HQQ', 'RTN', 'GPTQ'")

    return v


def _validate_weight_only_quant_config(v, values, field):
    if values.get("algorithm") is None:
        logger.debug("algorithm is not set, skip validation for weight_only_quant_configs")
        return v

    if v is None:
        v = {}

    config_keys = list(v.keys())
    if values["algorithm"] == "DEFAULT":
        default_config_keys = ["block_size", "is_symmetric", "accuracy_level"]
    elif values["algorithm"] == "RTN":
        default_config_keys = ["ratios"]
    elif values["algorithm"] == "HQQ":
        default_config_keys = ["block_size", "bits", "axis"]
    elif values["algorithm"] == "GPTQ":
        default_config_keys = ["percdamp", "block_size", "actorder", "mse", "perchannel"]

    if not all(key in default_config_keys for key in config_keys):
        invalid_config_keys = set(config_keys) - set(default_config_keys)
        logger.warning(
            "Invalid weight_only_quant_configs: %s for algorithm %s. Allowed keys are: %s",
            invalid_config_keys,
            values["algorithm"],
            default_config_keys,
        )
        v = {key: v[key] for key in default_config_keys if key in v}
    return v
