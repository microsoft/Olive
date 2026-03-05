# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import patch


class TestQuantizeFlow:
    @patch("olive.cli.init.onnx_flow._ask")
    def test_static_quantization_default_calib(self, mock_ask):
        from olive.cli.init.onnx_flow import QUANT_STATIC, _quantize_flow

        with patch("olive.cli.init.onnx_flow.prompt_calibration_source", return_value=None):
            mock_ask.return_value = QUANT_STATIC
            result = _quantize_flow("/model.onnx")
        cmd = result["command"]
        assert "--implementation ort" in cmd
        assert "--precision int8" in cmd

    @patch("olive.cli.init.onnx_flow._ask")
    def test_dynamic_quantization(self, mock_ask):
        from olive.cli.init.onnx_flow import QUANT_DYNAMIC, _quantize_flow

        mock_ask.return_value = QUANT_DYNAMIC
        result = _quantize_flow("/model.onnx")
        cmd = result["command"]
        assert "--algorithm rtn" in cmd
        assert "--implementation ort" in cmd

    @patch("olive.cli.init.onnx_flow._ask")
    def test_bnb_quantization(self, mock_ask):
        from olive.cli.init.onnx_flow import QUANT_BNB, _quantize_flow

        mock_ask.return_value = QUANT_BNB
        result = _quantize_flow("/model.onnx")
        cmd = result["command"]
        assert "--implementation bnb" in cmd
        assert "--precision nf4" in cmd

    @patch("olive.cli.init.onnx_flow.build_calibration_args", return_value=" -d data --split train --max_samples 128")
    @patch("olive.cli.init.onnx_flow.prompt_calibration_source")
    @patch("olive.cli.init.onnx_flow._ask")
    def test_static_with_calibration_data(self, mock_ask, mock_calib, mock_build):
        from olive.cli.init.onnx_flow import QUANT_STATIC, _quantize_flow
        from olive.cli.init.wizard import SOURCE_HF

        mock_ask.return_value = QUANT_STATIC
        mock_calib.return_value = {
            "source": SOURCE_HF,
            "data_name": "data",
            "subset": "",
            "split": "train",
            "num_samples": "128",
        }
        result = _quantize_flow("/model.onnx")
        cmd = result["command"]
        assert "--implementation ort" in cmd
        assert "-d data" in cmd


class TestOptimizeFlow:
    @patch("olive.cli.init.onnx_flow._ask")
    def test_generates_command(self, mock_ask):
        from olive.cli.init.onnx_flow import _optimize_flow

        mock_ask.side_effect = ["CPUExecutionProvider", "fp32"]
        result = _optimize_flow("/model.onnx")
        assert result["command"] == "olive optimize -m /model.onnx --provider CPUExecutionProvider --precision fp32"


class TestTuneSessionFlow:
    @patch("olive.cli.init.onnx_flow._ask")
    def test_cpu_with_options(self, mock_ask):
        from olive.cli.init.onnx_flow import _tune_session_flow

        mock_ask.side_effect = [
            "cpu",  # device
            ["CPUExecutionProvider"],  # providers
            "4",  # cpu_cores
            False,  # io_bind
            False,  # enable_cuda_graph
        ]
        result = _tune_session_flow("/model.onnx")
        cmd = result["command"]
        assert "--device cpu" in cmd
        assert "--providers_list CPUExecutionProvider" in cmd
        assert "--cpu_cores 4" in cmd
        assert "--io_bind" not in cmd

    @patch("olive.cli.init.onnx_flow._ask")
    def test_gpu_with_io_bind_and_cuda_graph(self, mock_ask):
        from olive.cli.init.onnx_flow import _tune_session_flow

        mock_ask.side_effect = [
            "gpu",  # device
            ["CUDAExecutionProvider"],  # providers
            "",  # cpu_cores (skip)
            True,  # io_bind
            True,  # enable_cuda_graph
        ]
        result = _tune_session_flow("/model.onnx")
        cmd = result["command"]
        assert "--device gpu" in cmd
        assert "--io_bind" in cmd
        assert "--enable_cuda_graph" in cmd


class TestConvertPrecisionFlow:
    def test_generates_command(self):
        from olive.cli.init.onnx_flow import _convert_precision_flow

        result = _convert_precision_flow("/model.onnx")
        assert result["command"] == "olive run-pass --pass-name OnnxFloatToFloat16 -m /model.onnx"


class TestGraphOptFlow:
    def test_generates_command(self):
        from olive.cli.init.onnx_flow import _graph_opt_flow

        result = _graph_opt_flow("/model.onnx")
        assert result["command"] == "olive optimize -m /model.onnx --precision fp32"


class TestRunOnnxFlowRouting:
    @patch("olive.cli.init.onnx_flow._optimize_flow")
    @patch("olive.cli.init.onnx_flow._ask_select")
    def test_routes_to_optimize(self, mock_select, mock_flow):
        from olive.cli.init.onnx_flow import OP_OPTIMIZE, run_onnx_flow

        mock_select.return_value = OP_OPTIMIZE
        mock_flow.return_value = {"command": "test"}
        run_onnx_flow({"model_path": "/m.onnx"})
        mock_flow.assert_called_once_with("/m.onnx")

    @patch("olive.cli.init.onnx_flow._quantize_flow")
    @patch("olive.cli.init.onnx_flow._ask_select")
    def test_routes_to_quantize(self, mock_select, mock_flow):
        from olive.cli.init.onnx_flow import OP_QUANTIZE, run_onnx_flow

        mock_select.return_value = OP_QUANTIZE
        mock_flow.return_value = {"command": "test"}
        run_onnx_flow({"model_path": "/m.onnx"})
        mock_flow.assert_called_once()

    @patch("olive.cli.init.onnx_flow._graph_opt_flow")
    @patch("olive.cli.init.onnx_flow._ask_select")
    def test_routes_to_graph_opt(self, mock_select, mock_flow):
        from olive.cli.init.onnx_flow import OP_GRAPH_OPT, run_onnx_flow

        mock_select.return_value = OP_GRAPH_OPT
        mock_flow.return_value = {"command": "test"}
        run_onnx_flow({"model_path": "/m.onnx"})
        mock_flow.assert_called_once_with("/m.onnx")

    @patch("olive.cli.init.onnx_flow._convert_precision_flow")
    @patch("olive.cli.init.onnx_flow._ask_select")
    def test_routes_to_convert_precision(self, mock_select, mock_flow):
        from olive.cli.init.onnx_flow import OP_CONVERT_PRECISION, run_onnx_flow

        mock_select.return_value = OP_CONVERT_PRECISION
        mock_flow.return_value = {"command": "test"}
        run_onnx_flow({"model_path": "/m.onnx"})
        mock_flow.assert_called_once()

    @patch("olive.cli.init.onnx_flow._tune_session_flow")
    @patch("olive.cli.init.onnx_flow._ask_select")
    def test_routes_to_tune_session(self, mock_select, mock_flow):
        from olive.cli.init.onnx_flow import OP_TUNE_SESSION, run_onnx_flow

        mock_select.return_value = OP_TUNE_SESSION
        mock_flow.return_value = {"command": "test"}
        run_onnx_flow({"model_path": "/m.onnx"})
        mock_flow.assert_called_once()

    @patch("olive.cli.init.onnx_flow._ask_select", return_value="unknown")
    def test_unknown_operation_returns_empty(self, mock_select):
        from olive.cli.init.onnx_flow import run_onnx_flow

        result = run_onnx_flow({"model_path": "/m.onnx"})
        assert not result
