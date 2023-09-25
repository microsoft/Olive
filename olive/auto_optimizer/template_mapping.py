# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from copy import deepcopy
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

TEMPLATE_CONFIG_PATH = Path(__file__).resolve().parent / "config_template"

model_template_mapping = {
    "bert": [["OnnxConversion", "OrtTransformersOptimization", "OnnxQuantization", "OrtPerfTuning"]],
    "open_llama": [["OptimumConversion", "OrtTransformersOptimization", "OptimumMerging"]],
}


def get_model_template_passes(model_type):
    # if cannot get model_type, remove None
    if not model_type:
        return None
    return deepcopy(model_template_mapping[model_type])


def get_model_template_config(model_type):
    model_config = None
    try:
        with (TEMPLATE_CONFIG_PATH / f"{model_type}.yaml").open() as f:
            model_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Cannot find template config for model type: {model_type}")
    return model_config
