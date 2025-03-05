# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Type, Union

import onnx
from packaging import version

from olive.common.config_utils import validate_config
from olive.common.pydantic_v1 import validator
from olive.common.utils import IntEnumBase, StrEnumBase, exclude_keys, hash_string
from olive.data.config import DataConfig
from olive.exception import OlivePassError
from olive.hardware.accelerator import AcceleratorSpec
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
from olive.search.search_parameter import Boolean, Categorical, Conditional, ConditionalDefault

logger = logging.getLogger(__name__)

# pylint: disable=consider-using-with

# common config for both static and dynamic quantization
_onnx_quantization_config = {
    "weight_type": PassConfigParam(
        type_=str,
        default_value="QInt8",
        search_defaults=Categorical(["QInt8", "QUInt8"]),
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
    "op_types_to_exclude": PassConfigParam(
        type_=list,
        default_value=None,
        description="""
            List of operator types to exclude from quantization. If None, all quantizable. op_types_to_quantize takes
            precedence over op_types_to_exclude. If both are set, op_types_to_quantize will be used.
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
        search_defaults=Boolean(),
        description="""
            Quantize weights per channel.
            Tips: When to use reduce_range and per-channel quantization:
            https://onnxruntime.ai/docs/performance/quantization.html#when-to-use-reduce-range-and-per-channel-quantization
        """,
    ),
    "reduce_range": PassConfigParam(
        type_=bool,
        default_value=False,
        search_defaults=Boolean(),
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
        search_defaults=Boolean(),
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
    "data_config": PassConfigParam(
        type_=Union[DataConfig, Dict],
        description="""
            Data config for calibration, required if quant_mode is 'static'
        """,
    ),
}

_static_optional_config = {
    "calibrate_method": PassConfigParam(
        type_=str,
        default_value="MinMax",
        search_defaults=Categorical(["MinMax", "Entropy", "Percentile"]),
        description="Supported calibration methods are MinMax, Entropy and Percentile.",
    ),
    "calibration_providers": PassConfigParam(
        type_=list,
        default_value=None,
        description="""
            Execution providers to run the session during calibration.
            Default is None which uses [ "CPUExecutionProvider" ].
        """,
    ),
    "quant_format": PassConfigParam(
        type_=str,
        default_value="QDQ",
        search_defaults=Categorical(["QOperator", "QDQ"]),
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
        search_defaults=Conditional(
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
    "qnn_extra_options": PassConfigParam(
        type_=dict,
        default_value=None,
        description="""
            Extra options for QNN quantization. Please refer to get_qnn_qdq_config at
            https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/execution_providers/qnn/quant_config.py.
            By default, the options are set to None. Options are only used if prepare_qnn_config is set to True.
        """,
    ),
}


def get_calibration_dataloader(config, model_path=None, io_config=None, calibration_providers=None):
    data_config = validate_config(config.data_config, DataConfig)
    return data_config.to_data_container().create_calibration_dataloader(
        model_path=model_path, io_config=io_config, calibration_providers=calibration_providers
    )


class OnnxQuantization(Pass):
    """Quantize ONNX model with static/dynamic quantization techniques."""

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
                search_defaults=Categorical(["dynamic", "static"]),
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
            if isinstance(value.search_defaults, Categorical):
                # ignore the parameter if quant_mode is dynamic
                # if quant_mode is static, use the search_defaults in static_optional_config by making it conditional
                value.search_defaults = Conditional(
                    parents=("quant_mode",),
                    support={("static",): value.search_defaults},
                    default=Conditional.get_ignored_choice(),
                )
            elif isinstance(value.search_defaults, Conditional):
                # ignore the parameter if quant_mode is dynamic
                # if quant_mode is static, use the search_defaults in static_optional_config by expanding the parents
                value.search_defaults = Conditional(
                    parents=("quant_mode", *value.search_defaults.parents),
                    support={
                        ("static", *key): value.search_defaults.support[key] for key in value.search_defaults.support
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

    @classmethod
    def validate_config(
        cls,
        config: Type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.quant_mode == "static":
            if (
                config.weight_type == "QInt8"
                and config.activation_type == "QInt8"
                and config.quant_format == "QOperator"
            ):
                # S8S8 with QOperator will be slow on x86-64 CPUs and should be avoided in general.
                # https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection
                # But we still allow it for users to try at their own risk. Olive just warns this to users.
                logger.warning(
                    "S8S8 with QOperator will be slow on x86-64 CPUs and should be avoided in general, try QDQ instead."
                )
            if config.EnableSubgraph is True:
                logger.info("EnableSubgraph is not supported for static quantization.")
                return False
        return True

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if model_has_adapters(model.model_path):
            logger.info("Model has adapters which should not be quantized. Returning the model without quantization.")
            return model

        from onnxruntime import __version__ as OrtVersion

        if version.parse(OrtVersion) < version.parse("1.18.0"):
            raise ValueError("Onnx Quantization is only supported for onnxruntime>=1.18.0")

        # use .release so that nightly releases are counted as above the version
        ort_less_than_1_21 = version.parse(OrtVersion).release < version.parse("1.21.0").release

        from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
        from onnxruntime.quantization.calibrate import CalibrationMethod

        # start with a copy of the config
        run_config = config.dict()
        is_static = run_config["quant_mode"] == "static"
        if is_static:
            assert config.data_config, "data_config is required for static quantization."

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # extra config
        extra_options = deepcopy(config.extra_options) or {}
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

        # keys not needed for quantization
        to_delete = [
            "quant_mode",
            "quant_preprocess",
            "prepare_qnn_config",
            "qnn_extra_options",
            "op_types_to_exclude",
            *_dataloader_config.keys(),
            *get_external_data_config().keys(),
        ]

        # update string values to enum values
        if is_static:
            run_config.update(
                {
                    "calibrate_method": CalibrationMethod[run_config["calibrate_method"]],
                    "quant_format": QuantFormat[run_config["quant_format"]],
                    "activation_type": QuantType[run_config["activation_type"]],
                    "weight_type": QuantType[run_config["weight_type"]],
                    "extra_options": extra_options,
                }
            )
            if ort_less_than_1_21:
                if run_config["calibration_providers"]:
                    logger.warning("calibration_providers is not supported for onnxruntime<1.21.0. It will be ignored.")
                to_delete += ["calibration_providers"]
        else:
            to_delete += list(_static_optional_config.keys())
            run_config.update(
                {
                    "weight_type": QuantType[run_config["weight_type"]],
                    "extra_options": extra_options,
                }
            )

        # remove keys not needed for quantization
        run_config = exclude_keys(run_config, to_delete)

        # there is no op_types_to_exclude in the quantizer, will exclude indirectly through nodes_to_exclude
        nodes_to_exclude = run_config["nodes_to_exclude"] or []
        if config.op_types_to_exclude:
            for node in onnx.load(model.model_path, load_external_data=False).graph.node:
                if node.op_type in config.op_types_to_exclude:
                    nodes_to_exclude.append(node.name)
        run_config["nodes_to_exclude"] = nodes_to_exclude

        # to be safe, run the quantizer with use_external_data_format set to `True` and
        # `model_output` to a temporary directory
        # reload the model and save to output_model_path using the external data config
        new_tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")
        tmp_model_path = str(Path(new_tmp_dir.name) / Path(output_model_path).name)

        if is_static:
            # get the dataloader
            dataloader = get_calibration_dataloader(
                config, model.model_path, model.io_config, config.calibration_providers
            )
            # TODO(anyone): generalize this option to prepare_qdq_config so that other NPU eps can use it
            # to call get_qdq_config
            if config.prepare_qnn_config:
                from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

                qnn_extra_options = config.qnn_extra_options or {}
                if init_overrides := _get_qnn_init_overrides(model, config):
                    qnn_extra_options["init_overrides"] = init_overrides

                # TODO(anyone): remove the version check once minimum version is 1.21.0
                additional_options = {}
                if not ort_less_than_1_21:
                    additional_options = {
                        "calibration_providers": run_config["calibration_providers"],
                        "op_types_to_quantize": run_config["op_types_to_quantize"],
                        "nodes_to_exclude": run_config["nodes_to_exclude"],
                    }

                qnn_config = get_qnn_qdq_config(
                    model_input=model.model_path,
                    calibration_data_reader=dataloader,
                    calibrate_method=run_config["calibrate_method"],
                    activation_type=run_config["activation_type"],
                    weight_type=run_config["weight_type"],
                    per_channel=run_config["per_channel"],
                    activation_symmetric=config.ActivationSymmetric,
                    weight_symmetric=config.WeightSymmetric,
                    **qnn_extra_options,
                    **additional_options,
                )
                # override the run_config with qnn_config
                # get all attributes of qnn_config
                run_config = {k: v for k, v in inspect.getmembers(qnn_config) if not k.startswith("_")}
                # remove the calibration_data_reader from run_config
                run_config = exclude_keys(run_config, ("calibration_data_reader", "use_external_data_format"))
                # TODO(jambayk): raise this > 1.21.0 if quantizer tool changes don't make it to 1.20.0
                if ort_less_than_1_21:
                    if nodes_to_exclude:
                        run_config["nodes_to_exclude"].extend(nodes_to_exclude)
                    if config.op_types_to_quantize:
                        run_config["op_types_to_quantize"] = config.op_types_to_quantize
                    elif config.op_types_to_exclude:
                        # op_types_to_quantize takes precedence over op_types_to_exclude
                        run_config["op_types_to_quantize"] = list(
                            set(run_config["op_types_to_quantize"]) - set(config.op_types_to_exclude)
                        )

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


class OnnxQuantizationPreprocess(Pass):
    """ONNX Quantization Preprocess Pass. Same as OnnxQuantization quant_preprocess."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "skip_optimization": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Skip model optimization step if true. This may result in ONNX shape"
                    " inference failure for some models."
                ),
            ),
            "skip_onnx_shape": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Skip ONNX shape inference. Symbolic shape inference is most effective"
                    " with transformer based models. Skipping all shape inferences may"
                    " reduce the effectiveness of quantization, as a tensor with unknown"
                    " shape can not be quantized."
                ),
            ),
            "skip_symbolic_shape": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Skip symbolic shape inference. Symbolic shape inference is most"
                    " effective with transformer based models. Skipping all shape"
                    " inferences may reduce the effectiveness of quantization, as a tensor"
                    " with unknown shape can not be quantized."
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime.quantization.preprocess import quant_pre_process

        with tempfile.TemporaryDirectory(dir=tempfile.tempdir, prefix="olive_tmp") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            tmp_model_path = resolve_onnx_path(tmp_dir_path)
            try:
                quant_pre_process(
                    input_model_path=model.model_path,
                    output_model_path=tmp_model_path,
                    auto_merge=True,
                    save_as_external_data=True,
                    skip_optimization=config.skip_optimization,
                    skip_onnx_shape=config.skip_onnx_shape,
                    skip_symbolic_shape=config.skip_symbolic_shape,
                    verbose=3,  # set verbose to 3 to get more information about the preprocessing
                )
            except Exception:
                logger.exception(
                    "Failed to run quantization preprocessing with error."
                    " Please retry with `skip_optimization = True` etc"
                )
                raise

            onnx_model = onnx.load(tmp_model_path)
            output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
            return model_proto_to_olive_model(onnx_model, output_model_path, config)


class OnnxDynamicQuantization(OnnxQuantization):
    """ONNX Dynamic Quantization Pass."""

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
            config["quant_format"].search_defaults = Categorical(["QDQ"])
            # Recently Int16/Uint16 is added into onnx runtime quantization only in QDQ mode.
            # for QNN EP integration, we give this workaround to support Int16/Uint16 in QDQ mode.
            # TODO(jiapli): remove this workaround once figure out the Int16/UInt16 in latest quantization
            config["activation_type"].search_defaults = Categorical(["QInt8", "QUInt8", "QUInt16", "QInt16"])
            config["weight_type"].search_defaults = Categorical(["QInt8", "QUInt8", "QUInt16", "QInt16"])
            config["prepare_qnn_config"].default_value = True
            # in QNN EP, the default value WeightSymmetric is None
            # but in base quantizer, the default value is True.
            config["WeightSymmetric"].default_value = None
        return config


class OnnxMatMul4Quantizer(Pass):
    """Quantize ONNX models' MatMul operations to 4-bit weights."""

    class AccuracyLevel(IntEnumBase):
        unset = 0
        fp32 = 1
        fp16 = 2
        bf16 = 3
        int8 = 4

    class Algorithm(StrEnumBase):
        DEFAULT = "DEFAULT"
        HQQ = "HQQ"
        RTN = "RTN"
        GPTQ = "GPTQ"

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
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
            "nodes_to_include": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of node names to include in quantization.",
            ),
            "op_types_to_quantize": PassConfigParam(
                type_=list,
                default_value=None,
                description=(
                    'List of operator types to quantize. Default value is None = ["MatMul"]. Supported op types'
                    " are: MatMul, Gather."
                ),
            ),
            "quant_axes": PassConfigParam(
                type_=Dict[str, int],
                default_value=None,
                description='op:axis, which axis to quantize for an op. Default is None = {"MatMul": 0, "Gather": 1}',
            ),
            "accuracy_level": PassConfigParam(
                type_=OnnxMatMul4Quantizer.AccuracyLevel,
                default_value=None,
                description=(
                    "Accuracy level of the 4-bit quantized MatMul computation. Refer to the MatMulNBits contrib op's"
                    " 'accuracy_level' attribute for details"
                    " (https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits)."
                ),
            ),
            "algorithm": PassConfigParam(
                type_=OnnxMatMul4Quantizer.Algorithm,
                default_value=None,
                description=(
                    "The algorithm used to quantize weight. If None, the default algorithm is used with quant"
                    " config created from the pass configuration."
                ),
            ),
            "weight_only_quant_configs": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "If 'algorithm' is provided and this is None, the config is constructed from the pass"
                    " configuration. If provided, the it takes precedence. Refer to"
                    " https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_4bits_quantizer.py"
                    " for details."
                ),
            ),
            **get_external_data_config(),
            # static_dataloder_config
            **deepcopy(_dataloader_config),
        }

    @classmethod
    def _validators(cls) -> Dict[str, Callable]:
        return {
            "validate_quant_config": validator("weight_only_quant_configs", allow_reuse=True)(
                _validate_weight_only_quant_config
            ),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime import __version__ as OrtVersion

        if version.parse(OrtVersion) < version.parse("1.18.0"):
            raise ValueError("MatMul4BitsQuantizer is only supported for onnxruntime>=1.18.0")

        from onnxruntime.quantization.matmul_4bits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            GPTQWeightOnlyQuantConfig,
            HQQWeightOnlyQuantConfig,
            MatMul4BitsQuantizer,
            RTNWeightOnlyQuantConfig,
        )

        algo_to_config = {
            "DEFAULT": DefaultWeightOnlyQuantConfig,
            "HQQ": HQQWeightOnlyQuantConfig,
            "RTN": RTNWeightOnlyQuantConfig,
            "GPTQ": GPTQWeightOnlyQuantConfig,
        }

        if model_has_adapters(model.model_path) and config.algorithm not in {None, "DEFAULT"}:
            logger.info(
                "Model has adapters which should only be quantized with algorithm=None or DEFAULT. Got %s. Returning"
                " the model without quantization.",
                config.algorithm,
            )
            return model

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        kwargs = {
            "block_size": config.block_size,
            "is_symmetric": config.is_symmetric,
            "nodes_to_exclude": config.nodes_to_exclude,
            "accuracy_level": config.accuracy_level,
        }
        for key in ["nodes_to_include", "op_types_to_quantize", "quant_axes"]:
            if value := getattr(config, key):
                if version.parse(OrtVersion) < version.parse("1.20.0"):
                    raise ValueError(f"MatMul4BitsQuantizer {key} is only supported for onnxruntime>=1.20.0")
                kwargs[key] = value

        if woq_config_class := algo_to_config.get(config.algorithm, None):
            algo_config = config.weight_only_quant_configs or {}
            for key in inspect.signature(woq_config_class.__init__).parameters:
                if key in algo_config:
                    if key in kwargs and kwargs[key] != algo_config[key]:
                        # algo config overrides pass config
                        logger.warning(
                            "The pass config parameter %s's value %s is different from the algorithm config's value %s."
                            " The algorithm config's value will be used.",
                            key,
                            kwargs[key],
                            algo_config[key],
                        )
                        kwargs[key] = algo_config[key]
                elif key in kwargs:
                    # get value from pass config
                    algo_config[key] = kwargs[key]
            if config.algorithm == "GPTQ":
                algo_config["calibration_data_reader"] = get_calibration_dataloader(config)
            kwargs["algo_config"] = woq_config_class(**algo_config)
        else:
            kwargs["algo_config"] = None

        quant = MatMul4BitsQuantizer(model.load_model(), **kwargs)
        quant.process()
        # topologically sort the graph at the end since previous optimizations may have broken it
        quant.model.topological_sort()
        # quant.model._check_init is not needed since it's only meant for float8 quantization

        # save the model to the output path and return the model
        return model_proto_to_olive_model(quant.model.model, output_model_path, config)


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
    else:
        raise ValueError(f"Unsupported algorithm: {values['algorithm']}")
    default_config_keys.extend(["op_types_to_quantize", "quant_axes"])

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


def _get_qnn_init_overrides(model_handler: ONNXModelHandler, config: Type[BasePassConfig]):
    # get qnn overrides from the input model
    model_attributes = model_handler.model_attributes or {}
    mp_init_overrides = model_attributes.get("mixed_precision_overrides") or {}
    init_overrides = {}
    config.qnn_extra_options = config.qnn_extra_options or {}
    if mp_init_overrides and "init_overrides" not in config.qnn_extra_options:
        from onnxruntime.quantization import QuantType

        # use QuantType to get the quantization type
        init_overrides = {
            tensor: [{"quant_type": QuantType.from_string(quant["quant_type"])} for quant in quant_types]
            for tensor, quant_types in mp_init_overrides.items()
        }
        # add `convert_outputs` to the TensorQuantOverridesHelper
        convert_outputs = config.convert_outputs or {}
        for output_name, output_convert_type in convert_outputs.items():
            init_overrides[output_name] = init_overrides.get(output_name, [{}])
            init_overrides[output_name][0]["quant_type"] = init_overrides[output_name][0].get(
                "quant_type"
            ) or QuantType.from_string(config.activation_type or "QUInt8")
            init_overrides[output_name][0]["convert"] = {"quant_type": QuantType.from_string(output_convert_type)}
    return init_overrides
