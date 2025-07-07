# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Union

import onnx
from packaging import version

from olive.common.config_utils import validate_config
from olive.common.utils import exclude_keys, hash_string
from olive.constants import Precision
from olive.data.config import DataConfig
from olive.exception import OlivePassError
from olive.hardware.accelerator import AcceleratorSpec
from olive.hardware.constants import ExecutionProvider
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
    "precision": PassConfigParam(
        type_=Precision,
        default_value=Precision.INT8,
        search_defaults=Categorical([Precision.INT8, Precision.UINT8]),
        description="""
            Data type for quantizing weights which is used both in dynamic
            and static quantization. 'int8' for signed 8-bit integer,
            'uint8' for unsigned 8-bit integer.
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
    "activation_symmetric": PassConfigParam(
        type_=bool,
        default_value=False,
        description="""
            Symmetric quantization for activations.
        """,
    ),
    "weight_symmetric": PassConfigParam(
        type_=bool,
        default_value=None,
        description="""
            Symmetric quantization for weights. Defaults to None. If set to None, it is assumed true if
            precision is signed, false otherwise.
        """,
    ),
}

_extra_options_config = {
    "extra_options": PassConfigParam(
        type_=dict,
        default_value=None,
        description="""
            Key value pair dictionary for `extra_options` in quantization. Please refer to
            https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py
            for details about the supported options. If an option is one of
            ActivationSymmetric, WeightSymmetric, MinimumRealRange or TensorQuantOverrides, it will be overwritten
            by the corresponding config parameter value.
        """,
    ),
}

# static quantization specific config
_dataloader_config = {
    "data_config": PassConfigParam(
        type_=Union[DataConfig, dict],
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
        type_=Precision,
        default_value=Precision.INT8,
        # the search space is conditional on quant_format and precision
        # the equivalent joint search space for (quant_format, precision, activation) is
        #   {
        #       (QDQ, Precision.INT8, Precision.INT8),
        #       (QDQ, Precision.UINT8, Precision.UINT8),
        #       (QOperator, Precision.UINT8, Precision.UINT8),
        #   }
        search_defaults=Conditional(
            parents=("quant_format", "precision"),
            support={
                ("QDQ", Precision.INT8): Categorical([Precision.INT8]),
                ("QDQ", Precision.UINT8): Categorical([Precision.UINT8]),
                ("QOperator", Precision.UINT8): Categorical([Precision.UINT8]),
                # invalid choice for QOperator, Precision.INT8
                ("QOperator", Precision.INT8): Conditional.get_invalid_choice(),
            },
        ),
        description="""
            Quantization data type of activation. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        """,
    ),
    "min_real_range": PassConfigParam(
        type_=float,
        default_value=None,
        description="""
            Minimum real range for quantization. If set, enforces the minimum range between rmin and rmax.
        """,
    ),
    "tensor_quant_overrides": PassConfigParam(
        type_=dict,
        default_value=None,
        description="""
            tensor-level quantization overrides.
        """,
    ),
    "prepare_qdq_config": PassConfigParam(
        type_=bool,
        default_value=True,
        description="""
            Generate a quantization configuration for a full integer QDQ model. Otherwise, only a limited set of
            operators are quantized. Only supported after onnxruntime 1.21.0 for EPs other than QNN.
        """,
    ),
}


def quant_type_from_precision(p):
    from onnxruntime.quantization import QuantType

    mapping = {
        Precision.INT4: QuantType["QInt4"],
        Precision.UINT4: QuantType["QUInt4"],
        Precision.INT8: QuantType["QInt8"],
        Precision.UINT8: QuantType["QUInt8"],
        Precision.INT16: QuantType["QInt16"],
        Precision.UINT16: QuantType["QUInt16"],
    }
    return mapping.get(p)


def get_calibration_dataloader(config, model_path=None, io_config=None, calibration_providers=None):
    data_config = validate_config(config.data_config, DataConfig)
    return data_config.to_data_container().create_calibration_dataloader(
        model_path=model_path, io_config=io_config, calibration_providers=calibration_providers
    )


# extra options name: (param_name, use in dynamic quantization)
_param_extra_options_mapping = {
    "ActivationSymmetric": ("activation_symmetric", True),
    "WeightSymmetric": ("weight_symmetric", True),
    "MinimumRealRange": ("min_real_range", False),
    "TensorQuantOverrides": ("tensor_quant_overrides", False),
}


class OnnxQuantization(Pass):
    """Quantize ONNX model with static/dynamic quantization techniques."""

    def _initialize(self):
        super()._initialize()
        # pylint: disable=attribute-defined-outside-init
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
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
        config.update(deepcopy(_extra_options_config))

        # external data config
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

        if not quant_type_from_precision(config.precision):
            logger.warning("Unsupported precision: %s", config.precision)
            return False

        if config.quant_mode == "static":
            if not quant_type_from_precision(config.activation_type):
                logger.warning("Unsupported activation_type: %s", config.activation_type)
                return False

            if (
                config.precision == Precision.INT8
                and config.activation_type == Precision.INT8
                and config.quant_format == "QOperator"
            ):
                # S8S8 with QOperator will be slow on x86-64 CPUs and should be avoided in general.
                # https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection
                # But we still allow it for users to try at their own risk. Olive just warns this to users.
                logger.warning(
                    "S8S8 with QOperator will be slow on x86-64 CPUs and should be avoided in general, try QDQ instead."
                )
        return True

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if model_has_adapters(model.model_path):
            logger.info("Model has adapters which should not be quantized. Returning the model without quantization.")
            return model

        from onnxruntime import __version__ as OrtVersion

        if version.parse(OrtVersion) < version.parse("1.18.0"):
            raise ValueError("Onnx Quantization is only supported for onnxruntime>=1.18.0")

        # use .release so that nightly releases are counted as above the version
        ort_less_than_1_21 = version.parse(OrtVersion).release < version.parse("1.21.0").release

        from onnxruntime.quantization import QuantFormat, quantize_dynamic, quantize_static
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
        intersection = set(extra_options.keys()).intersection(set(_param_extra_options_mapping.keys()))
        if intersection:
            logger.warning(
                "Extra config keys %s are already exposed in the pass config. They will be overwritten by"
                " the corresponding pass config parameter values.",
                intersection,
            )
        for key, (value, use_in_dynamic) in _param_extra_options_mapping.items():
            if run_config.get(value) is not None and (is_static or use_in_dynamic):
                # add the value to extra_options
                extra_options[key] = run_config[value]
            # remove the key from run_config
            run_config.pop(value, None)

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
            "prepare_qdq_config",
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
                    "activation_type": run_config["activation_type"],
                    "precision": run_config["precision"],
                    "extra_options": extra_options,
                }
            )
            if ort_less_than_1_21:
                if run_config["calibration_providers"]:
                    logger.warning("calibration_providers is not supported for onnxruntime<1.21.0. It will be ignored.")
                to_delete += ["calibration_providers"]
        else:
            to_delete += list(_static_optional_config.keys())
            to_delete += ["precision"]
            run_config.update(
                {
                    "weight_type": quant_type_from_precision(run_config["precision"]),
                    "extra_options": extra_options,
                }
            )

        # remove keys not needed for quantization
        run_config = exclude_keys(run_config, to_delete)

        # there is no op_types_to_exclude in the quantizer, will exclude indirectly through nodes_to_exclude
        run_config["nodes_to_exclude"] = nodes_to_exclude = run_config.get("nodes_to_exclude") or []
        if config.op_types_to_exclude:
            for node in onnx.load(model.model_path, load_external_data=False).graph.node:
                if node.op_type in config.op_types_to_exclude:
                    nodes_to_exclude.append(node.name)

        # to be safe, run the quantizer with use_external_data_format set to `True` and
        # `model_output` to a temporary directory
        # reload the model and save to output_model_path using the external data config
        new_tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")
        tmp_model_path = str(Path(new_tmp_dir.name) / Path(output_model_path).name)

        if is_static:
            run_config = self.get_static_run_config(model, config, run_config, ort_less_than_1_21)
            try:
                quantize_static(
                    model_input=model.model_path,
                    model_output=tmp_model_path,
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

    def get_static_run_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], run_config: dict, ort_less_than_1_21: bool
    ) -> dict:
        """Prepare the run config for static quantization."""
        dataloader = get_calibration_dataloader(config, model.model_path, model.io_config, config.calibration_providers)
        if config.quant_format != "QDQ" or not config.prepare_qdq_config:
            run_config.update(
                {
                    "calibration_data_reader": dataloader,
                    "use_external_data_format": True,
                    "weight_type": quant_type_from_precision(run_config["precision"]),
                    "activation_type": quant_type_from_precision(run_config["activation_type"]),
                }
            )
            run_config.pop("precision", None)
            return run_config

        is_qnn_ep = self.accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider

        if is_qnn_ep:
            from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config as get_qdq_config
        else:
            if ort_less_than_1_21:
                raise ValueError("prepare_qdq_config is only supported for onnxruntime>=1.21.0.")
            from onnxruntime.quantization.quantize import get_qdq_config

        get_qdq_config_kwargs = {
            "model_input": model.model_path,
            "calibration_data_reader": dataloader,
            "weight_type": quant_type_from_precision(run_config["precision"]),
            "activation_type": quant_type_from_precision(run_config["activation_type"]),
            "calibrate_method": run_config["calibrate_method"],
            "per_channel": run_config["per_channel"],
        }
        if not ort_less_than_1_21:
            # only available in onnxruntime>=1.21.0 for prepare_qnn_config
            for key in ["calibration_providers", "op_types_to_quantize", "nodes_to_exclude"]:
                get_qdq_config_kwargs[key] = run_config[key]
        # put the exposed extra options in the get_qdq_config_kwargs
        extra_options = deepcopy(run_config["extra_options"])
        for key, (qdq_key, _) in _param_extra_options_mapping.items():
            if key in extra_options:
                get_qdq_config_kwargs[qdq_key] = extra_options[key]
                # remove the key from extra_options
                extra_options.pop(key, None)
        if extra_options:
            get_qdq_config_kwargs["extra_options"] = extra_options
        # tensor_quant_overrides is init_overrides in qnn
        if is_qnn_ep:
            if "tensor_quant_overrides" in get_qdq_config_kwargs:
                get_qdq_config_kwargs["init_overrides"] = get_qdq_config_kwargs.pop("tensor_quant_overrides")
            elif init_overrides := _get_qnn_init_overrides(model, config):
                get_qdq_config_kwargs["init_overrides"] = init_overrides
            if "min_real_range" in get_qdq_config_kwargs:
                # min_real_range is not supported in QNN EP which enforces 1e-4
                get_qdq_config_kwargs.pop("min_real_range")

        # get the qdq config
        qdq_config = get_qdq_config(**get_qdq_config_kwargs)

        # override the run_config with qdq_config
        new_run_config = {k: v for k, v in inspect.getmembers(qdq_config) if not k.startswith("_")}
        # always run with use_external_data_format
        new_run_config["use_external_data_format"] = True
        if ort_less_than_1_21:
            # only available in onnxruntime>=1.21.0 for prepare_qnn_config
            if run_config["nodes_to_exclude"]:
                new_run_config["nodes_to_exclude"].extend(run_config["nodes_to_exclude"])
            if config.op_types_to_quantize:
                new_run_config["op_types_to_quantize"] = config.op_types_to_quantize
            elif config.op_types_to_exclude:
                # op_types_to_quantize takes precedence over op_types_to_exclude
                new_run_config["op_types_to_quantize"] = list(
                    set(new_run_config["op_types_to_quantize"]) - set(config.op_types_to_exclude)
                )

        return new_run_config


class OnnxQuantizationPreprocess(Pass):
    """ONNX Quantization Preprocess Pass. Same as OnnxQuantization quant_preprocess."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
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
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
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
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        if accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider:
            raise ValueError("QNNExecutionProvider is not supported for dynamic quantization.")
        config = {
            "quant_mode": PassConfigParam(type_=str, default_value="dynamic", description="dynamic quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_onnx_quantization_config))
        config.update(deepcopy(_extra_options_config))
        # external data config
        config.update(get_external_data_config())
        return config


class OnnxStaticQuantization(OnnxQuantization):
    """ONNX Static Quantization Pass."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "quant_mode": PassConfigParam(type_=str, default_value="static", description="static quantization mode")
        }
        # common quantization config
        config.update(deepcopy(_onnx_quantization_config))
        # static quantization specific config
        config.update(deepcopy(_dataloader_config))
        config.update(deepcopy(_static_optional_config))
        # exposed extra options config
        config.update(deepcopy(_extra_options_config))
        # external data config
        config.update(get_external_data_config())
        if accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider:
            config["quant_format"].search_defaults = Categorical(["QDQ"])
            # Recently Int16/Uint16 is added into onnx runtime quantization only in QDQ mode.
            # for QNN EP integration, we give this workaround to support Int16/Uint16 in QDQ mode.
            # TODO(jiapli): remove this workaround once figure out the Int16/UInt16 in latest quantization
            config["activation_type"].search_defaults = Categorical(
                [Precision.INT8, Precision.UINT8, Precision.INT16, Precision.UINT16]
            )
            config["precision"].search_defaults = Categorical(
                [Precision.INT8, Precision.UINT8, Precision.INT16, Precision.UINT16]
            )
        return config

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Override this method to return False by using the accelerator spec information."""
        return False


def _get_qnn_init_overrides(model_handler: ONNXModelHandler, config: type[BasePassConfig]):
    # get qnn overrides from the input model
    model_attributes = model_handler.model_attributes or {}
    mp_init_overrides = model_attributes.get("mixed_precision_overrides") or {}
    init_overrides = {}
    if mp_init_overrides:
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
            ) or quant_type_from_precision(config.activation_type or Precision.UINT8)
            init_overrides[output_name][0]["convert"] = {"quant_type": QuantType.from_string(output_convert_type)}
    return init_overrides
