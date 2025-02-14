# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
from test.unit_test.utils import ONNX_MODEL_PATH, get_onnx_model
from unittest.mock import MagicMock, patch

import pytest
from onnxruntime.transformers.fusion_options import FusionOptions

from olive.hardware import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, DEFAULT_GPU_TRT_ACCELERATOR
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.passes.onnx.common import get_external_data_config
from olive.passes.onnx.transformer_optimization import OrtTransformersOptimization

# pylint: disable=redefined-outer-name, abstract-method, protected-access


def test_fusion_options():
    config = {"model_type": "bart", "optimization_options": {"use_multi_head_attention": True}}
    config = OrtTransformersOptimization.generate_config(DEFAULT_CPU_ACCELERATOR, config, disable_search=True)
    transformer_optimization = OrtTransformersOptimization(DEFAULT_CPU_ACCELERATOR, config, True)
    run_config = config.dict()
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


@pytest.mark.parametrize("use_gpu", [True, False])
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize(
    "accelerator_spec", [DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, DEFAULT_GPU_TRT_ACCELERATOR]
)
@pytest.mark.parametrize("mock_inferece_session", [True, False])
def test_invalid_ep_config(use_gpu, fp16, accelerator_spec, mock_inferece_session, tmp_path, caplog):
    import onnxruntime as ort
    from onnxruntime.transformers.onnx_model import OnnxModel
    from packaging import version

    if accelerator_spec == DEFAULT_GPU_TRT_ACCELERATOR and not mock_inferece_session:
        pytest.skip("Skipping test: TRT EP does not support compiled nodes when mock_inferece_session=False")

    logger = logging.getLogger("olive")
    logger.propagate = True

    input_model = get_onnx_model()
    config = {"model_type": "bert", "use_gpu": use_gpu, "float16": fp16}
    config = OrtTransformersOptimization.generate_config(accelerator_spec, config, disable_search=True)
    p = OrtTransformersOptimization(accelerator_spec, config, True)
    is_pruned = not p.validate_config(config, accelerator_spec)

    if accelerator_spec.execution_provider == "CPUExecutionProvider":
        if fp16 and use_gpu:
            assert is_pruned
            assert (
                "CPUExecutionProvider does not support float16 very well, please avoid to use float16." in caplog.text
            )
        elif use_gpu:
            assert is_pruned
            assert "CPUExecutionProvider does not support GPU inference, please avoid to use use_gpu." in caplog.text

    if accelerator_spec.execution_provider == "TensorrtExecutionProvider" and fp16:
        assert is_pruned
        assert (
            "TensorRT has its own float16 implementation, please avoid to use float16 in transformers "
            "optimization. Suggest to set 'trt_fp16_enable' as True in OrtSessionParamsTuning." in caplog.text
        )

    if not is_pruned:
        inference_session_mock_call_count = 0

        def inference_session_init(
            self,
            path_or_bytes,
            sess_options=None,
            providers=None,
            provider_options=None,
            **kwargs,
        ):
            nonlocal inference_session_mock_call_count
            inference_session_mock_call_count += 1
            shutil.copyfile(ONNX_MODEL_PATH, sess_options.optimized_model_filepath)

        with patch("onnxruntime.transformers.optimizer.optimize_by_fusion") as optimize_by_fusion_mock:
            optimize_by_fusion_mock.return_value = OnnxModel(input_model.load_model())
            output_folder = str(tmp_path / "onnx")
            if mock_inferece_session:
                with patch.object(ort.InferenceSession, "__init__", new=inference_session_init):
                    p.run(input_model, output_folder)
            else:
                p.run(input_model, output_folder)
            optimize_by_fusion_mock.assert_called()

        if accelerator_spec.execution_provider == "TensorrtExecutionProvider":
            if accelerator_spec.execution_provider not in ort.get_available_providers():
                if use_gpu:
                    # the use_gpu will be ignored by optimize_model, please refef to the following links for more info.
                    # https://github.com/microsoft/onnxruntime/blob/v1.15.1/onnxruntime/python/tools/transformers/optimizer.py#L280
                    if version.parse(ort.__version__) >= version.parse("1.16.0"):
                        # for TensorRT EP, the graph optimization will be skipped but the fusion will be applied.
                        assert "There is no gpu for onnxruntime to do optimization." in caplog.text
                        if mock_inferece_session:
                            assert inference_session_mock_call_count == 0
                else:
                    # for cpu graph optimization, the graph optimization will always be run. So there is not need check
                    if mock_inferece_session:
                        assert inference_session_mock_call_count > 0
            else:
                if mock_inferece_session:
                    assert inference_session_mock_call_count > 0


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
