# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy

from onnxruntime.transformers.fusion_options import FusionOptions

from olive.passes.onnx import OrtTransformersOptimization
from olive.passes.onnx.common import get_external_data_config


def test_fusion_options():
    config = {"model_type": "bart", "optimization_options": {"use_multi_head_attention": True}}
    config_class, config = OrtTransformersOptimization.generate_search_space(config, True)
    transformer_optimization = OrtTransformersOptimization(config_class, config)
    run_config = deepcopy(config)
    del (
        run_config["float16"],
        run_config["input_int32"],
        run_config["keep_io_types"],
        run_config["force_fp32_ops"],
        run_config["target_provider"],
    )
    for key in get_external_data_config():
        del run_config[key]
    transformer_optimization._set_fusion_options(run_config)
    olive_fusion_options = run_config["optimization_options"]

    ort_fusion_options = FusionOptions("bart")
    assert vars(olive_fusion_options) != vars(ort_fusion_options)

    ort_fusion_options.use_multi_head_attention = True
    assert vars(olive_fusion_options) == vars(ort_fusion_options)
