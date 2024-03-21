# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_glue_huggingface_data_config

import pytest
import yaml

from olive.auto_optimizer import AutoOptimizer, AutoOptimizerConfig
from olive.auto_optimizer.template_mapping import get_pass_flows_by_accelerator_ep_precision
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, DEFAULT_GPU_TRT_ACCELERATOR
from olive.model import ModelConfig

# pylint: disable=attribute-defined-outside-init


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
        self.data_configs = {"__input_model_data_config__": get_glue_huggingface_data_config()}

    @pytest.mark.parametrize(
        ("accelerator_spec", "auto_optimizer_config", "expected_cuda_fp16", "expected_trt_fp16"),
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
        ],
    )
    def test_regulate_fp16(self, accelerator_spec, auto_optimizer_config, expected_cuda_fp16, expected_trt_fp16):
        metrics = [get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, goal_type="max-degradation")]
        evaluator_config = OliveEvaluatorConfig(metrics=metrics)
        auto_optimizer = AutoOptimizer(
            input_model_config=self.input_model_config,
            evaluator_config=evaluator_config,
            accelerator_spec=accelerator_spec,
            auto_optimizer_config=auto_optimizer_config,
            data_configs=self.data_configs,
        )

        pass_config, _ = auto_optimizer.suggest()
        trans_opt_name = "OrtTransformerOptimization_cuda_fp16" if expected_cuda_fp16 else "OrtTransformersOptimization"
        perf_opt_name = "OrtPerfTuning_trt_fp16" if expected_trt_fp16 else "OrtPerfTuning"
        assert pass_config[trans_opt_name]["config"]["float16"] == expected_cuda_fp16
        assert pass_config[perf_opt_name]["config"]["enable_cuda_graph"] == expected_cuda_fp16
        assert pass_config[perf_opt_name]["config"]["trt_fp16_enable"] == expected_trt_fp16

    @pytest.mark.parametrize(
        ("metrics_configs", "accelerator_spec", "auto_optimizer_config", "expected_pass_flows"),
        [
            (
                [{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "max-degradation"}}],
                DEFAULT_CPU_ACCELERATOR,
                None,
                [
                    ["OnnxConversion", "OrtTransformersOptimization", "OrtPerfTuning"],
                    ["OnnxConversion", "OrtTransformersOptimization", "OnnxQuantization", "OrtPerfTuning"],
                    ["OnnxConversion", "OrtTransformersOptimization", "IncQuantization", "OrtPerfTuning"],
                ],
            ),
            (
                # cannot tolerate accuracy drop, then skip quantization
                [
                    {
                        "args": [AccuracySubType.ACCURACY_SCORE],
                        "kwargs": {"goal_type": "max-degradation", "goal_value": 0},
                    }
                ],
                DEFAULT_CPU_ACCELERATOR,
                AutoOptimizerConfig(precisions=["fp32"]),
                [["OnnxConversion", "OrtTransformersOptimization", "OrtPerfTuning"]],
            ),
            (
                # running on gpu-cuda, skip quantization
                [{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "max-degradation"}}],
                DEFAULT_GPU_CUDA_ACCELERATOR,
                AutoOptimizerConfig(precisions=["fp16"]),
                [
                    ["OnnxConversion", "OrtTransformerOptimization_cuda_fp16", "OrtPerfTuning"],
                    ["OnnxConversion", "OrtTransformersOptimization", "OrtMixedPrecision", "OrtPerfTuning"],
                ],
            ),
        ],
    )
    def test_regulate_pass(self, metrics_configs, accelerator_spec, auto_optimizer_config, expected_pass_flows):
        metrics = [get_accuracy_metric(*mc["args"], **mc["kwargs"]) for mc in metrics_configs]
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
        assert sorted(pass_flows) == sorted(expected_pass_flows)

    def test_pass_flows_generation_opt_level_0(self):
        pass_flows_map = Path(__file__).parent / "mock_data" / "available_pass_flows.yaml"
        with pass_flows_map.open() as f:
            pass_flows_map = yaml.safe_load(f)["mapping"]

        for k, pf in pass_flows_map.items():
            k_list = k.split("_")
            accelerator, ep, precision = k_list[0], k_list[1], k_list[2]
            rls_pf = get_pass_flows_by_accelerator_ep_precision(0, accelerator, ep, precision)
            assert sorted(rls_pf) == sorted(pf)

    def test_pass_config_when_no_evaluator(self):
        auto_optimizer = AutoOptimizer(
            input_model_config=self.input_model_config,
            evaluator_config=None,
            accelerator_spec=DEFAULT_CPU_ACCELERATOR,
            auto_optimizer_config=None,
            data_configs=self.data_configs,
        )
        pass_config, _ = auto_optimizer.suggest()
        for pass_name in pass_config:
            assert pass_config[pass_name]["disable_search"] is True
