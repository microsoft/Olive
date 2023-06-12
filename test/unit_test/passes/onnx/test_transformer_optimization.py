# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from copy import deepcopy
from pathlib import Path
from test.unit_test.utils import get_onnx_model

from onnxruntime.transformers.fusion_options import FusionOptions

from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.onnx import OrtTransformersOptimization
from olive.passes.onnx.common import get_external_data_config
from olive.systems.local import LocalSystem


def test_fusion_options():
    config = {"model_type": "bart", "optimization_options": {"use_multi_head_attention": True}}
    config = OrtTransformersOptimization.generate_search_space(DEFAULT_CPU_ACCELERATOR, config, True)
    transformer_optimization = OrtTransformersOptimization(DEFAULT_CPU_ACCELERATOR, config, True)
    run_config = deepcopy(config)
    del (
        run_config["float16"],
        run_config["input_int32"],
        run_config["keep_io_types"],
        run_config["force_fp32_ops"],
    )
    for key in get_external_data_config():
        del run_config[key]
    transformer_optimization._set_fusion_options(run_config)
    olive_fusion_options = run_config["optimization_options"]

    ort_fusion_options = FusionOptions("bart")
    assert vars(olive_fusion_options) != vars(ort_fusion_options)

    ort_fusion_options.use_multi_head_attention = True
    assert vars(olive_fusion_options) == vars(ort_fusion_options)


def test_ort_transformer_optimization_pass():
    # setup
    local_system = LocalSystem()
    input_model = get_onnx_model()
    config = {"model_type": "bert"}

    config = OrtTransformersOptimization.generate_search_space(DEFAULT_CPU_ACCELERATOR, config, disable_search=True)
    p = OrtTransformersOptimization(DEFAULT_CPU_ACCELERATOR, config, True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        local_system.run_pass(p, input_model, output_folder)
