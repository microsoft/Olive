#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
from pathlib import Path

import onnx

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.vitis_ai.estimate_npu_latency import EstimateNPULatency
from test.utils import get_onnx_model


class TestEstimateNPULatency:
    """Test cases for EstimateNPULatency pass."""

    def test_estimate_latency_basic(self, tmp_path):
        """Test Perf Estimator call with automatic Olive version."""
        # Setup
        input_model = get_onnx_model()
        config = {}
        p = create_pass_from_dict(EstimateNPULatency, config, disable_search=True)
        output_folder = str(tmp_path / "onnx")

        # Execute
        output_model = p.run(input_model, output_folder)

        # Assert we created output csv for latency results
        estimates_csv = f"{os.path.dirname(input_model.model_path)}/concise_summary"
        assert Path(estimates_csv).exists()

        # Assert
        assert Path(output_model.model_path).exists()
        # Load the output model and check graph name
        onnx_model = onnx.load_model(output_model.model_path)
        assert onnx_model.graph.name == "main_graph"
