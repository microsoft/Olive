# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
from unittest.mock import MagicMock, patch

import pytest
from onnxruntime.transformers.fusion_options import FusionOptions

from olive.hardware import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, DEFAULT_GPU_TRT_ACCELERATOR
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.passes.onnx.common import get_external_data_config
from olive.passes.onnx.transformer_optimization import OrtTransformersOptimization
from test.utils import ONNX_MODEL_PATH, get_onnx_model

# pylint: disable=redefined-outer-name, abstract-method, protected-access


def test_fusion_options():
    config = {"model_type": "bart", "optimization_options": {"use_multi_head_attention": True}}
    config = OrtTransformersOptimization.generate_config(DEFAULT_CPU_ACCELERATOR, config, disable_search=True)
    transformer_optimization = OrtTransformersOptimization(DEFAULT_CPU_ACCELERATOR, config, True)
    run_config = config.model_dump()
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

    config = OrtTransformersOptimization.generate_config(DEFAULT_CPU_ACCELERATOR, config, disable_search=True)
    p = OrtTransformersOptimization(DEFAULT_CPU_ACCELERATOR, config, True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, output_folder)


@pytest.mark.parametrize(
    ("use_gpu", "fp16", "accelerator_spec", "expected_message"),
    [
        pytest.param(
            False, True, DEFAULT_CPU_ACCELERATOR, "CPUExecutionProvider does not support float16", id="cpu-fp16"
        ),
        pytest.param(True, False, DEFAULT_CPU_ACCELERATOR, "CPUExecutionProvider does not support GPU", id="cpu-gpu"),
        pytest.param(
            False, True, DEFAULT_GPU_TRT_ACCELERATOR, "TensorRT has its own float16 implementation", id="trt-fp16"
        ),
        pytest.param(True, True, DEFAULT_GPU_CUDA_ACCELERATOR, None, id="cuda-valid"),
    ],
)
def test_invalid_ep_config(use_gpu, fp16, accelerator_spec, expected_message, caplog):
    config = {"model_type": "bert", "use_gpu": use_gpu, "float16": fp16}

    with caplog.at_level(logging.INFO, logger="olive"):
        pass_config = OrtTransformersOptimization.generate_config(accelerator_spec, config, disable_search=True)
        p = OrtTransformersOptimization(accelerator_spec, pass_config, True)
        is_valid = p.validate_config(pass_config, accelerator_spec)

    assert is_valid is (expected_message is None)
    if expected_message:
        assert expected_message in caplog.text


def test_transformer_optimization_valid_cuda_config_runs(tmp_path):
    import onnxruntime as ort
    from onnxruntime.transformers.onnx_model import OnnxModel

    input_model = get_onnx_model()
    config = {"model_type": "bert", "use_gpu": True, "float16": True}
    pass_config = OrtTransformersOptimization.generate_config(DEFAULT_GPU_CUDA_ACCELERATOR, config, disable_search=True)
    p = OrtTransformersOptimization(DEFAULT_GPU_CUDA_ACCELERATOR, pass_config, True)

    def inference_session_init(
        self,
        path_or_bytes,
        sess_options=None,
        providers=None,
        provider_options=None,
        **kwargs,
    ):
        shutil.copyfile(ONNX_MODEL_PATH, sess_options.optimized_model_filepath)

    with (
        patch("onnxruntime.transformers.optimizer.optimize_by_fusion") as optimize_by_fusion_mock,
        patch.object(ort.InferenceSession, "__init__", new=inference_session_init),
    ):
        optimize_by_fusion_mock.return_value = OnnxModel(input_model.load_model())
        output_model = p.run(input_model, str(tmp_path / "onnx"))

    optimize_by_fusion_mock.assert_called_once()
    assert output_model.model_path


def test_transformer_optimization_invalid_model_type(tmp_path):
    input_model = get_onnx_model()
    config = {"model_type": None}

    config = OrtTransformersOptimization.generate_config(DEFAULT_CPU_ACCELERATOR, config, disable_search=True)
    p = OrtTransformersOptimization(DEFAULT_CPU_ACCELERATOR, config, True)
    output_folder = str(tmp_path / "onnx")

    output = p.run(input_model, output_folder)

    assert output == input_model


@patch("onnxruntime.transformers.optimizer.optimize_model")
@patch("olive.passes.onnx.transformer_optimization.model_proto_to_olive_model")
@patch("onnxruntime.get_available_providers", MagicMock(return_value=["DmlExecutionProvider"]))
@patch("onnxruntime.__version__", "1.17.0")
def test_optimization_with_provider(mock_proto_to_model, mock_optimize_model, tmp_path):
    input_model = get_onnx_model()
    config = {"model_type": "bert", "use_gpu": True}

    dml_ep = AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="DmlExecutionProvider")
    config = OrtTransformersOptimization.generate_config(dml_ep, config, disable_search=True)
    p = OrtTransformersOptimization(dml_ep, config, True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, output_folder)
    assert mock_optimize_model.call_args.kwargs["provider"] == "dml"
