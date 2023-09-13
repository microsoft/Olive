# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from test.unit_test.utils import get_onnx_model

import pytest
from onnxruntime.transformers.fusion_options import FusionOptions

from olive.hardware import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, DEFAULT_GPU_TRT_ACCELERATOR
from olive.passes.onnx import OrtTransformersOptimization
from olive.passes.onnx.common import get_external_data_config


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


def test_ort_transformer_optimization_pass(tmp_path):
    # setup
    input_model = get_onnx_model()
    config = {"model_type": "bert"}

    config = OrtTransformersOptimization.generate_search_space(DEFAULT_CPU_ACCELERATOR, config, disable_search=True)
    p = OrtTransformersOptimization(DEFAULT_CPU_ACCELERATOR, config, True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize(
    "accelerator_spec", [DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, DEFAULT_GPU_TRT_ACCELERATOR]
)
def test_invalid_ep_config(use_gpu, fp16, accelerator_spec, tmp_path):
    input_model = get_onnx_model()
    config = {"model_type": "bert", "use_gpu": use_gpu, "float16": fp16}
    config = OrtTransformersOptimization.generate_search_space(accelerator_spec, config, disable_search=True)
    p = OrtTransformersOptimization(accelerator_spec, config, True)
    is_pruned = not p.validate_search_point(config, accelerator_spec)
    if accelerator_spec.execution_provider == "CPUExecutionProvider":
        if use_gpu:
            assert is_pruned, "CPUExecutionProvider does not support GPU inference, please avoid to use use_gpu."
        if fp16:
            assert is_pruned, "CPUExecutionProvider does not support float16 very well, please avoid to use float16."

    if fp16 and accelerator_spec.execution_provider == "TensorrtExecutionProvider":
        assert is_pruned, (
            "TensorRT has its own float16 implementation, please avoid to use float16 in transformers "
            "optimization. Suggest to set 'trt_fp16_enable' as True in OrtPerfTuning."
        )

    if not is_pruned:
        output_folder = str(tmp_path / "onnx")
        p.run(input_model, None, output_folder)
