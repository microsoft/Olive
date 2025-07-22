# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
from argparse import ArgumentParser, Namespace
from unittest.mock import Mock, patch

from olive.cli.optimize import OptimizeCommand
from olive.constants import Precision
from olive.hardware.constants import ExecutionProvider


class TestOptimizeCommand:
    def test_register_subcommand(self):
        """Test that the optimize command registers properly."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        
        # This should not raise any exception
        OptimizeCommand.register_subcommand(subparsers)
        
        # Parse help to ensure command is registered
        with pytest.raises(SystemExit):  # argparse raises SystemExit on --help
            parser.parse_args(["optimize", "--help"])

    def test_basic_argument_parsing(self):
        """Test basic argument parsing for the optimize command."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        OptimizeCommand.register_subcommand(subparsers)
        
        # Test with minimal required arguments
        args = parser.parse_args([
            "optimize",
            "--model_path", "test_model",
            "--output_path", "output"
        ])
        
        assert args.model_path == "test_model"
        assert args.output_path == "output"
        assert args.provider == ExecutionProvider.CPUExecutionProvider.value
        assert args.device == "cpu"
        assert args.precision == Precision.FP32.value

    def test_all_arguments_parsing(self):
        """Test parsing with all optional arguments."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        OptimizeCommand.register_subcommand(subparsers)
        
        args = parser.parse_args([
            "optimize",
            "--model_path", "test_model",
            "--output_path", "output",
            "--provider", "CUDAExecutionProvider",
            "--device", "gpu",
            "--precision", "int4",
            "--act_precision", "int8",
            "--num_split", "2",
            "--memory", "8192",
            "--exporter", "dynamo_exporter",
            "--dim_param", "batch_size", "seq_len",
            "--dim_value", "1", "128",
            "--use_qdq_format",
            "--surgeries", "surgery1", "surgery2"
        ])
        
        assert args.provider == "CUDAExecutionProvider"
        assert args.device == "gpu"
        assert args.precision == "int4"
        assert args.act_precision == "int8"
        assert args.num_split == 2
        assert args.memory == 8192
        assert args.exporter == "dynamo_exporter"
        assert args.dim_param == ["batch_size", "seq_len"]
        assert args.dim_value == [1, 128]
        assert args.use_qdq_format is True
        assert args.surgeries == ["surgery1", "surgery2"]

    def test_device_provider_validation(self):
        """Test device and provider compatibility validation."""
        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="DmlExecutionProvider",
            device="cpu",  # Should be auto-adjusted to gpu
            precision="fp32",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            log_level=None
        )
        
        with patch('olive.cli.optimize.get_input_model_config') as mock_get_input:
            mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}
            
            command = OptimizeCommand(ArgumentParser(), args)
            command._validate_device_provider_compatibility()
            
            # Device should be auto-adjusted to gpu for DML provider
            assert command.args.device == "gpu"

    def test_precision_to_bits_conversion(self):
        """Test precision to bits conversion."""
        args = Namespace()
        command = OptimizeCommand(ArgumentParser(), args)
        
        assert command._precision_to_bits(Precision.INT4) == 4
        assert command._precision_to_bits(Precision.INT8) == 8
        assert command._precision_to_bits(Precision.INT16) == 16
        assert command._precision_to_bits(Precision.INT32) == 32

    @patch('olive.cli.optimize.get_input_model_config')
    def test_pass_scheduling_hf_model_gptq(self, mock_get_input):
        """Test pass scheduling for HF model with GPTQ quantization."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}
        
        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="int4",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            log_level=None
        )
        
        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)
        
        # Should include GPTQ pass for HF model with int4 precision
        assert "gptq" in passes_config
        assert passes_config["gptq"]["type"] == "GptqQuantizer"
        assert passes_config["gptq"]["bits"] == 4

    @patch('olive.cli.optimize.get_input_model_config')
    def test_pass_scheduling_onnx_model(self, mock_get_input):
        """Test pass scheduling for ONNX model."""
        mock_get_input.return_value = {"type": "OnnxModel", "model_path": "test_model.onnx"}
        
        args = Namespace(
            model_path="test_model.onnx",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="int4",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            log_level=None
        )
        
        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=False)
        
        # Should include OnnxBlockWiseRtnQuantization for ONNX model with int4
        assert "onnx_blockwise_rtn_quantization" in passes_config
        assert passes_config["onnx_blockwise_rtn_quantization"]["type"] == "OnnxBlockWiseRtnQuantization"

    @patch('olive.cli.optimize.get_input_model_config')
    def test_pass_scheduling_qnn_provider(self, mock_get_input):
        """Test pass scheduling for QNN execution provider."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}
        
        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="QNNExecutionProvider",
            device="npu",
            precision="int4",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=["batch_size"],
            dim_value=[1],
            use_qdq_format=False,
            surgeries=None,
            log_level=None
        )
        
        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)
        
        # Should include QuaRot, DynamicToFixedShape, StaticLLM, and EPContextBinaryGenerator
        assert "quarot" in passes_config
        assert "dynamic_to_fixed_shape" in passes_config
        assert "static_llm" in passes_config
        assert "ep_context_binary_generator" in passes_config

    @patch('olive.cli.optimize.get_input_model_config')
    def test_pass_scheduling_fp16_precision(self, mock_get_input):
        """Test pass scheduling for FP16 precision."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}
        
        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="fp16",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            log_level=None
        )
        
        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)
        
        # Should include OnnxFloatToFloat16 for FP16 precision
        assert "onnx_float_to_float16" in passes_config

    @patch('olive.cli.optimize.get_input_model_config')
    def test_pass_scheduling_with_surgeries(self, mock_get_input):
        """Test pass scheduling with graph surgeries."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}
        
        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="fp32",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=["surgery1", "surgery2"],
            log_level=None
        )
        
        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)
        
        # Should include GraphSurgeries pass
        assert "graph_surgeries" in passes_config
        assert passes_config["graph_surgeries"]["surgeries"] == ["surgery1", "surgery2"]