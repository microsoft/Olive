# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_accuracy_metric, get_glue_huggingface_data_config

import pytest

from olive.auto_optimizer import AutoOptimizer, AutoOptimizerConfig
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, DEFAULT_GPU_TRT_ACCELERATOR
from olive.model import ModelConfig


class TestAutoOptimizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.input_model_config = ModelConfig(
            type="PyTorchModel",
            config={
                "hf_config": {
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task": "text-classification",
                }
            },
        )
        metrics = [get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, goal_type="max-degradation")]
        self.evaluator_config = OliveEvaluatorConfig(metrics=metrics)
        self.data_configs = {"__input_model_data_config__": get_glue_huggingface_data_config()}

    @pytest.mark.parametrize(
        "accelerator_spec, auto_optimizer_config, expected_cuda_fp16, expected_trt_fp16",
        [
            (
                # running on gpu-cuda, enable cuda fp16, disable trt fp16
                DEFAULT_GPU_CUDA_ACCELERATOR,
                None,
                True,
                False,
            ),
            (
                # running on gpu-trt, disable cuda fp16, enable trt fp16
                DEFAULT_GPU_TRT_ACCELERATOR,
                None,
                False,
                True,
            ),
            (
                # running on gpu-cuda, enable global precision
                DEFAULT_GPU_CUDA_ACCELERATOR,
                AutoOptimizerConfig(precisions=["fp32"]),
                False,
                False,
            ),
        ],
    )
    def test_regulate_fp16(self, accelerator_spec, auto_optimizer_config, expected_cuda_fp16, expected_trt_fp16):
        auto_optimizer = AutoOptimizer(
            input_model_config=self.input_model_config,
            evaluator_config=self.evaluator_config,
            accelerator_spec=accelerator_spec,
            auto_optimizer_config=auto_optimizer_config,
            data_configs=self.data_configs,
        )

        pass_config, pass_flows = auto_optimizer.suggest()
        assert len(next(iter(pass_flows))) == 3
        assert pass_config["OrtTransformersOptimization"]["float16"] == expected_cuda_fp16
        assert pass_config["OrtPerfTuning"]["enable_cuda_graph"] == expected_cuda_fp16
        assert pass_config["OrtPerfTuning"]["trt_fp16_enable"] == expected_trt_fp16

    @pytest.mark.parametrize(
        "metrics, accelerator_spec, auto_optimizer_config, expected_pass_flows",
        [
            (
                [get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, goal_type="max-degradation")],
                DEFAULT_CPU_ACCELERATOR,
                None,
                [["OnnxConversion", "OrtTransformersOptimization", "OnnxQuantization", "OrtPerfTuning"]],
            ),
            (
                # cannot tolerate accuracy drop, then skip quantization
                [get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, goal_type="max-degradation", goal_value=0)],
                DEFAULT_CPU_ACCELERATOR,
                None,
                [["OnnxConversion", "OrtTransformersOptimization", "OrtPerfTuning"]],
            ),
            (
                # running on gpu-cuda, skip quantization
                [get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, goal_type="max-degradation")],
                DEFAULT_GPU_CUDA_ACCELERATOR,
                None,
                [["OnnxConversion", "OrtTransformersOptimization", "OrtPerfTuning"]],
            ),
            (
                # running on gpu-cuda, skip quantization
                [get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, goal_type="max-degradation")],
                DEFAULT_GPU_CUDA_ACCELERATOR,
                AutoOptimizerConfig(disabled_passes=["OrtPerfTuning"]),
                [["OnnxConversion", "OrtTransformersOptimization"]],
            ),
        ],
    )
    def test_regulate_pass(self, metrics, accelerator_spec, auto_optimizer_config, expected_pass_flows):
        evaluator_config = OliveEvaluatorConfig(metrics=metrics)
        auto_optimizer = AutoOptimizer(
            input_model_config=self.input_model_config,
            evaluator_config=evaluator_config,
            accelerator_spec=accelerator_spec,
            auto_optimizer_config=auto_optimizer_config,
            data_configs=self.data_configs,
        )

        pass_config, pass_flows = auto_optimizer.suggest()
        assert pass_config, "Expect pass_config to be populated by auto optimizer"
        assert pass_flows == expected_pass_flows


class TestAutoOptimizerConfig:
    def test_pass_conflicts(self):
        with pytest.raises(ValueError):
            AutoOptimizerConfig(
                customized_pass_config={"OnnxConversion": {"target_opset": 15}}, disabled_passes=["OnnxConversion"]
            )
