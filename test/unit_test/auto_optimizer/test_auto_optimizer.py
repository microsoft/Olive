# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest

from olive.auto_optimizer import AutoOptimizer, AutoOptimizerConfig
from olive.constants import Precision
from olive.hardware import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, AcceleratorSpec, Device
from olive.model import ModelConfig

# pylint: disable=attribute-defined-outside-init


class TestAutoOptimizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_config = ModelConfig(
            type="PyTorchModel",
            config={
                "hf_config": {
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task": "text-classification",
                }
            },
        )

    @pytest.mark.parametrize(
        ("optimizer_config", "expected_pass_types"),
        [
            (
                AutoOptimizerConfig(
                    precision=Precision.FP16,
                    accelerator=DEFAULT_CPU_ACCELERATOR,
                    finetune=False,
                ),
                [
                    ["CaptureSplitInfo"],
                    ["QuaRot", "SpinQuant"],
                    ["ModelBuilder"],
                    ["OnnxIODataTypeConverter"],
                    ["MatMulNBitsToQDQ"],
                    ["SplitModel"],
                    ["ExtractAdapters"],
                ],
            ),
            (
                AutoOptimizerConfig(
                    precision=Precision.FP16,
                    accelerator=DEFAULT_GPU_CUDA_ACCELERATOR,
                    finetune=False,
                    use_model_builder=False,
                ),
                [
                    ["CaptureSplitInfo"],
                    ["QuaRot", "SpinQuant"],
                    ["OnnxConversion"],
                    ["OnnxPeepholeOptimizer"],
                    ["OrtTransformersOptimization"],
                    ["OnnxIODataTypeConverter"],
                    ["MatMulNBitsToQDQ"],
                    ["SplitModel"],
                    ["ExtractAdapters"],
                ],
            ),
            (
                AutoOptimizerConfig(
                    precision=Precision.FP32,
                    accelerator=DEFAULT_CPU_ACCELERATOR,
                    finetune=False,
                ),
                [
                    ["CaptureSplitInfo"],
                    ["QuaRot", "SpinQuant"],
                    ["ModelBuilder"],
                    ["OnnxIODataTypeConverter"],
                    ["MatMulNBitsToQDQ"],
                    ["SplitModel"],
                    ["ExtractAdapters"],
                ],
            ),
            (
                AutoOptimizerConfig(
                    precision=Precision.FP16,
                    accelerator=DEFAULT_GPU_CUDA_ACCELERATOR,
                    finetune=False,
                ),
                [
                    ["CaptureSplitInfo"],
                    ["QuaRot", "SpinQuant"],
                    ["ModelBuilder"],
                    ["OnnxIODataTypeConverter"],
                    ["MatMulNBitsToQDQ"],
                    ["SplitModel"],
                    ["ExtractAdapters"],
                ],
            ),
            (
                AutoOptimizerConfig(
                    precision=Precision.INT4,
                    accelerator=DEFAULT_GPU_CUDA_ACCELERATOR,
                    finetune=False,
                ),
                [
                    ["CaptureSplitInfo"],
                    ["QuaRot", "SpinQuant", "AutoAWQQuantizer", "GptqQuantizer"],
                    ["ModelBuilder"],
                    ["OnnxIODataTypeConverter"],
                    ["NVModelOptQuantization"],
                    ["MatMulNBitsToQDQ"],
                    ["SplitModel"],
                    ["ExtractAdapters"],
                ],
            ),
            (
                AutoOptimizerConfig(
                    precision=Precision.INT4,
                    accelerator=DEFAULT_GPU_CUDA_ACCELERATOR,
                    finetune=False,
                    quantize=False,
                ),
                [
                    ["CaptureSplitInfo"],
                    ["ModelBuilder"],
                    ["OnnxIODataTypeConverter"],
                    ["MatMulNBitsToQDQ"],
                    ["SplitModel"],
                    ["ExtractAdapters"],
                ],
            ),
            (
                AutoOptimizerConfig(
                    precision=Precision.INT4,
                    accelerator=AcceleratorSpec(accelerator_type=Device.NPU, execution_provider="QNNExecutionProvider"),
                    finetune=False,
                ),
                [
                    ["CaptureSplitInfo"],
                    ["QuaRot", "SpinQuant", "AutoAWQQuantizer", "GptqQuantizer"],
                    ["ModelBuilder"],
                    ["OnnxIODataTypeConverter"],
                    ["QNNPreprocess"],
                    ["MatMulNBitsToQDQ"],
                    ["SplitModel"],
                    ["ExtractAdapters"],
                ],
            ),
        ],
    )
    def test_generate_run_passes_configs(self, optimizer_config, expected_pass_types):
        auto_optimizer = AutoOptimizer(model_config=self.model_config, optimizer_config=optimizer_config)
        pass_configs = auto_optimizer.generate_run_passes_configs()
        assert pass_configs, "Expect pass_configs to be populated by auto optimizer"

        actual_pass_types = [[pc.type.lower() for pc in pcs] for pcs in pass_configs.values()]
        expected_pass_types = [[pt.lower() for pt in pts] for pts in expected_pass_types]
        assert actual_pass_types == expected_pass_types
