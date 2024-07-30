# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Callable, Dict, Optional

from olive.common.hf.utils import get_feature_from_task, get_model_config, get_tokenizer
from olive.common.utils import get_attr

if TYPE_CHECKING:
    from transformers.onnx import OnnxConfig

logger = logging.getLogger(__name__)


# patched version of transformers.onnx.features.supported_features_mapping
# to support additional models in olive
def patched_supported_features_mapping(
    *supported_features: str, onnx_config_cls: Optional[str] = None
) -> Dict[str, Callable]:
    """Generate the mapping between supported the features and their corresponding OnnxConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        onnx_config_cls: The OnnxConfig full name corresponding to the model.

    Returns:
        The dictionary mapping a feature to an OnnxConfig constructor.

    """
    if onnx_config_cls is None:
        raise ValueError("A OnnxConfig class must be provided")

    from olive.common.hf import onnx_config

    config_cls = get_attr(onnx_config, onnx_config_cls)
    mapping = {}
    for feature in supported_features:
        if "-with-past" in feature:
            mapping[feature] = partial(config_cls.with_past, task=feature.replace("-with-past", ""))
        else:
            mapping[feature] = partial(config_cls.from_model_config, task=feature)

    return mapping


# TODO(jambayk): switch to optimum backend and make this an optional feature
# remove "feature" entirely from the codebase
def get_onnx_config(model_name: str, task: str, feature: Optional[str] = None, **kwargs) -> "OnnxConfig":
    # pylint: disable=protected-access
    from transformers.onnx import FeaturesManager

    from olive.common.hf.onnx_config import ADDITIONAL_MODEL_TYPES

    # patch FeaturesManager._SUPPORTED_MODEL_TYPE to support additional models in olive
    for model_type, feature_list in ADDITIONAL_MODEL_TYPES.items():
        if model_type in FeaturesManager._SUPPORTED_MODEL_TYPE:
            continue
        # TODO(trajep): remove the need for unpacking feature_list
        features, onnx_config_cls = feature_list
        FeaturesManager._SUPPORTED_MODEL_TYPE[model_type] = patched_supported_features_mapping(
            *features, onnx_config_cls=onnx_config_cls
        )

    # if feature is not provided, try to get it from task
    # else use "default"
    feature = feature or get_feature_from_task(task) or "default"

    # don't want to load the model here since all we need is the config
    # model loading is expensive computationally and memory-wise for large models
    config = get_model_config(model_name, **kwargs)
    # recreate the logic for FeaturesManager.check_supported_model_or_raise to get the model_onnx_config
    # https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L712
    model_type = config.model_type.replace("_", "-")
    onnx_config = None
    try:
        model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
        if feature in model_features:
            onnx_config = FeaturesManager.get_config(model_type, feature)(config)
        else:
            logger.debug(
                "%s doesn't support feature %s. Supported features are: %s", model_type, feature, model_features
            )
    except KeyError:
        logger.debug("Model type %s is not supported", model_type)

    return onnx_config


def get_model_io_config(model_name: str, task: str, feature: Optional[str] = None, **kwargs):
    # just log a debug message if io_config is not found
    # this is not a critical error and the caller may not need the io_config
    model_config = get_onnx_config(model_name, task, feature, **kwargs)
    if not model_config:
        return None

    inputs = model_config.inputs
    outputs = model_config.outputs
    if not inputs or not outputs:
        # just log a warning and return None, since this is not a critical error
        # and following pass may not use the io_config, like OptimumConversion
        logger.debug("No inputs or outputs found from hf onnx_config %s. Won't use it to get io config", model_config)
        return None

    io_config = {}
    io_config["input_names"] = list(inputs.keys())
    io_config["output_names"] = list(outputs.keys())
    io_config["dynamic_axes"] = dict(chain(inputs.items(), outputs.items()))
    return io_config


def get_model_dummy_input(model_name: str, task: str, feature: Optional[str] = None, **kwargs):
    model_config = get_onnx_config(model_name, task, feature, **kwargs)
    if not model_config:
        return None
    tokenizer = get_tokenizer(model_name)
    return model_config.generate_dummy_inputs(tokenizer, framework="pt")
