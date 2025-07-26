# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser, Namespace
from unittest.mock import patch

import pytest

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
        args = parser.parse_args(["optimize", "--model_path", "test_model", "--output_path", "output"])

        assert args.model_path == "test_model"
        assert args.output_path == "output"
        assert args.provider == ExecutionProvider.CPUExecutionProvider.value
        assert args.device == "cpu"
        assert args.precision == Precision.FP32.value
        assert args.modality == "text"  # Default modality

    def test_all_arguments_parsing(self):
        """Test parsing with all optional arguments."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        OptimizeCommand.register_subcommand(subparsers)

        args = parser.parse_args(
            [
                "optimize",
                "--model_path",
                "test_model",
                "--output_path",
                "output",
                "--provider",
                "CUDAExecutionProvider",
                "--device",
                "gpu",
                "--precision",
                "int4",
                "--act_precision",
                "int8",
                "--num_split",
                "2",
                "--memory",
                "8192",
                "--exporter",
                "dynamo_exporter",
                "--dim_param",
                "batch_size",
                "seq_len",
                "--dim_value",
                "1",
                "128",
                "--use_qdq_format",
                "--surgeries",
                "surgery1",
                "surgery2",
                "--block_size",
                "64",
                "--modality",
                "vision",
            ]
        )

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
        assert args.block_size == 64
        assert args.modality == "vision"

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
            block_size=None,
            modality="text",
            log_level=None,
        )

        with patch("olive.cli.optimize.get_input_model_config") as mock_get_input:
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

    @patch("olive.cli.optimize.get_input_model_config")
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
            block_size=None,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        # Should include GPTQ pass for HF model with int4 precision
        assert "gptq" in passes_config
        assert passes_config["gptq"]["type"] == "Gptq"
        assert passes_config["gptq"]["bits"] == 4

    @patch("olive.cli.optimize.get_input_model_config")
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
            block_size=None,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=False)

        # Should include OnnxBlockWiseRtnQuantization for ONNX model with int4
        assert "onnx_blockwise_rtn_quantization" in passes_config
        assert passes_config["onnx_blockwise_rtn_quantization"]["type"] == "OnnxBlockWiseRtnQuantization"

    @patch("olive.cli.optimize.get_input_model_config")
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
            block_size=None,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        # Should include QuaRot, DynamicToFixedShape, StaticLLM, and EPContextBinaryGenerator
        assert "quarot" in passes_config
        assert "dynamic_to_fixed_shape" in passes_config
        assert "static_llm" in passes_config
        assert "ep_context_binary_generator" in passes_config

    @patch("olive.cli.optimize.get_input_model_config")
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
            block_size=None,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        # Should include OnnxFloatToFloat16 for FP16 precision
        assert "onnx_float_to_float16" in passes_config

    @patch("olive.cli.optimize.get_input_model_config")
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
            block_size=None,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        # Should include GraphSurgeries pass
        assert "graph_surgeries" in passes_config
        assert passes_config["graph_surgeries"]["surgeries"] == ["surgery1", "surgery2"]

    def test_block_size_argument_parsing(self):
        """Test block_size argument parsing."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        OptimizeCommand.register_subcommand(subparsers)

        # Test with positive block_size
        args = parser.parse_args(["optimize", "--model_path", "test_model", "--block_size", "128"])
        assert args.block_size == 128

        # Test with -1 block_size (per-channel)
        args = parser.parse_args(["optimize", "--model_path", "test_model", "--block_size", "-1"])
        assert args.block_size == -1

        # Test without block_size (should be None)
        args = parser.parse_args(["optimize", "--model_path", "test_model"])
        assert args.block_size is None

    @patch("olive.cli.optimize.get_input_model_config")
    def test_block_size_in_model_builder_pass(self, mock_get_input):
        """Test block_size usage in ModelBuilder pass."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}

        # Test with custom block_size
        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="int4",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter="model_builder",
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            block_size=64,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "model_builder" in passes_config
        assert passes_config["model_builder"]["int4_block_size"] == 64

    @patch("olive.cli.optimize.get_input_model_config")
    def test_block_size_minus_one_in_model_builder(self, mock_get_input):
        """Test block_size -1 in ModelBuilder pass (should default to 32)."""
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
            exporter="model_builder",
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            block_size=-1,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "model_builder" in passes_config
        # Should default to 32 since ModelBuilder doesn't support per-channel
        assert passes_config["model_builder"]["int4_block_size"] == 32

    @patch("olive.cli.optimize.get_input_model_config")
    def test_block_size_in_onnx_blockwise_pass(self, mock_get_input):
        """Test block_size usage in OnnxBlockWiseRtnQuantization pass."""
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
            block_size=256,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=False)

        assert "onnx_blockwise_rtn_quantization" in passes_config
        assert passes_config["onnx_blockwise_rtn_quantization"]["block_size"] == 256

    @patch("olive.cli.optimize.get_input_model_config")
    def test_block_size_minus_one_in_onnx_blockwise_pass(self, mock_get_input):
        """Test block_size -1 in OnnxBlockWiseRtnQuantization pass (per-channel)."""
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
            block_size=-1,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=False)

        assert "onnx_blockwise_rtn_quantization" in passes_config
        assert passes_config["onnx_blockwise_rtn_quantization"]["block_size"] == -1
        assert passes_config["onnx_blockwise_rtn_quantization"]["axis"] == 0

    @patch("olive.cli.optimize.get_input_model_config")
    def test_block_size_minus_one_in_static_quantization(self, mock_get_input):
        """Test block_size -1 enables per_channel in OnnxStaticQuantization."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}

        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="int8",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            block_size=-1,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "onnx_static_quantization" in passes_config
        assert passes_config["onnx_static_quantization"]["per_channel"] is True

    @patch("olive.cli.optimize.get_input_model_config")
    def test_block_size_invalid_value_model_builder(self, mock_get_input):
        """Test invalid block_size value gets adjusted to closest valid value for ModelBuilder."""
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
            exporter="model_builder",
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            block_size=100,  # Not a valid ModelBuilder block size
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "model_builder" in passes_config
        # Should be adjusted to the closest valid value (128)
        assert passes_config["model_builder"]["int4_block_size"] == 128

    @patch("olive.cli.optimize.get_input_model_config")
    def test_block_size_in_gptq_pass(self, mock_get_input):
        """Test block_size usage in Gptq pass as group_size."""
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
            block_size=64,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "gptq" in passes_config
        assert passes_config["gptq"]["group_size"] == 64

    @patch("olive.cli.optimize.get_input_model_config")
    def test_block_size_minus_one_in_gptq_pass(self, mock_get_input):
        """Test block_size -1 in Gptq pass (per-channel quantization)."""
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
            block_size=-1,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "gptq" in passes_config
        assert passes_config["gptq"]["group_size"] == -1

    def test_modality_argument_parsing(self):
        """Test modality argument parsing."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        OptimizeCommand.register_subcommand(subparsers)

        # Test default modality (text)
        args = parser.parse_args(["optimize", "--model_path", "test_model"])
        assert args.modality == "text"

        # Test explicit text modality
        args = parser.parse_args(["optimize", "--model_path", "test_model", "--modality", "text"])
        assert args.modality == "text"

        # Test vision modality
        args = parser.parse_args(["optimize", "--model_path", "test_model", "--modality", "vision"])
        assert args.modality == "vision"

    @patch("olive.cli.optimize.get_input_model_config")
    def test_modality_text_adds_data_config(self, mock_get_input):
        """Test that text modality adds data_config to OnnxStaticQuantization."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}

        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="int8",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            block_size=None,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "onnx_static_quantization" in passes_config
        config = passes_config["onnx_static_quantization"]
        assert "data_configs" in config
        assert len(config["data_configs"]) == 1

        data_config = config["data_configs"][0]
        assert data_config["name"] == "wikitext2_train"
        assert data_config["type"] == "HuggingfaceContainer"
        assert data_config["load_dataset_config"]["data_name"] == "wikitext"
        assert data_config["load_dataset_config"]["subset"] == "wikitext-2-raw-v1"
        assert data_config["load_dataset_config"]["split"] == "train"

        pre_process_config = data_config["pre_process_data_config"]
        assert pre_process_config["strategy"] == "line-by-line"
        assert pre_process_config["add_special_tokens"] is False
        assert pre_process_config["max_samples"] == 128
        assert pre_process_config["max_seq_len"] == 512

    @patch("olive.cli.optimize.get_input_model_config")
    def test_modality_vision_no_data_config(self, mock_get_input):
        """Test that vision modality does not add data_config to OnnxStaticQuantization."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}

        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="int8",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            block_size=None,
            modality="vision",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "onnx_static_quantization" in passes_config
        config = passes_config["onnx_static_quantization"]
        assert "data_configs" not in config

    @patch("olive.cli.optimize.get_input_model_config")
    def test_modality_text_with_block_size_and_data_config(self, mock_get_input):
        """Test that text modality works correctly with block_size -1 and adds data_config."""
        mock_get_input.return_value = {"type": "HfModel", "model_path": "test_model"}

        args = Namespace(
            model_path="test_model",
            output_path="output",
            provider="CPUExecutionProvider",
            device="cpu",
            precision="int8",
            act_precision=None,
            num_split=None,
            memory=None,
            exporter=None,
            dim_param=None,
            dim_value=None,
            use_qdq_format=False,
            surgeries=None,
            block_size=-1,
            modality="text",
            log_level=None,
        )

        command = OptimizeCommand(ArgumentParser(), args)
        passes_config = command._build_passes_config(is_hf_model=True)

        assert "onnx_static_quantization" in passes_config
        config = passes_config["onnx_static_quantization"]

        # Should have both per_channel and data_configs
        assert config["per_channel"] is True
        assert "data_configs" in config
        assert len(config["data_configs"]) == 1
