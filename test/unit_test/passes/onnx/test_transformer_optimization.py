# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from copy import deepcopy
from pathlib import Path
from test.unit_test.utils import get_onnx_model

from onnxruntime.transformers.fusion_options import FusionOptions

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OrtTransformersOptimization
from olive.passes.onnx.common import get_external_data_config
from olive.systems.local import LocalSystem


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


def test_ort_transformer_optimization_pass():
    # setup
    local_system = LocalSystem()
    input_model = get_onnx_model()
    config = {"model_type": "bert"}
    p = create_pass_from_dict(OrtTransformersOptimization, config, disable_search=True)
    with tempfile.TemporaryDirectory() as tempdir:
        output_folder = str(Path(tempdir) / "onnx")

        # execute
        local_system.run_pass(p, input_model, output_folder)
