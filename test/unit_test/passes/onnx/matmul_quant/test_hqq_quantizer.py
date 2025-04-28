# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
import torch
from onnx import GraphProto, NodeProto, TensorProto

from olive.passes.onnx.matmul_quant.hqq_quantizer import HQQWeightOnlyQuantConfig, HQQWeightOnlyQuantizer
from olive.passes.onnx.matmul_quant.utils import MSFT_DOMAIN, Algorithm, OpType

# pylint: disable=redefined-outer-name


@pytest.fixture
def default_config():
    return HQQWeightOnlyQuantConfig(block_size=128, bits=4, axis=1, op_types_to_quantize=None, quant_axes=None)


@pytest.fixture
def bits2_config():
    return HQQWeightOnlyQuantConfig(block_size=128, bits=2, axis=1, op_types_to_quantize=None, quant_axes=None)


@pytest.fixture
def axis0_config():
    return HQQWeightOnlyQuantConfig(block_size=128, bits=4, axis=0, op_types_to_quantize=None, quant_axes=None)


class TestHQQWeightOnlyQuantConfig:
    def test_init_default_values(self):
        config = HQQWeightOnlyQuantConfig()
        assert config.block_size == 128
        assert config.bits == 4
        assert config.axis == 1
        assert config.algorithm == Algorithm.HQQ

    def test_init_custom_values(self):
        config = HQQWeightOnlyQuantConfig(
            block_size=64, bits=2, axis=0, op_types_to_quantize=("MatMul",), quant_axes=(("MatMul", 0),)
        )
        assert config.block_size == 64
        assert config.bits == 2
        assert config.axis == 0
        assert config.algorithm == Algorithm.HQQ
        assert config.op_types_to_quantize == {"MatMul"}
        assert config.quant_axes == {"MatMul": 0}


class TestHQQWeightOnlyQuantizer:
    def test_init(self, default_config):
        quantizer = HQQWeightOnlyQuantizer(default_config)
        assert quantizer.config == default_config

    def test_optimize_weights(self):
        # Setup
        tensor = torch.randn(64, 128)
        scale = torch.ones(64, 1)
        zero = torch.zeros(64, 1)
        min_max = [0, 15]  # 4-bit range

        # Execute
        opt_scale, opt_zero = HQQWeightOnlyQuantizer.optimize_weights(tensor, scale, zero, min_max, axis=1)

        # Assert
        assert opt_scale.shape == scale.shape
        assert opt_zero.shape == zero.shape
        assert opt_scale.dtype == torch.float32  # CPU uses float32
        assert opt_zero.dtype == torch.float32
        assert opt_scale.device.type == "cpu"
        assert opt_zero.device.type == "cpu"

    def test_pack_on_row_fast_248bit_4bit(self):
        # Setup
        ori_int_tensor = torch.randint(0, 16, (128, 64), dtype=torch.int)
        pack_tensor = torch.zeros((128, 32), dtype=torch.uint8)

        # Execute
        HQQWeightOnlyQuantizer.pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits=4)

        # Assert
        # Check some random samples to verify packing logic
        # For 4-bit packing, each byte contains 2 values
        # Lower 4 bits: first value, Upper 4 bits: second value
        for _ in range(10):  # Check a few random positions
            row = np.random.randint(0, 128)
            col = np.random.randint(0, 32)
            unpacked_col = col * 2

            # Lower 4 bits should be the first value
            assert (pack_tensor[row, col] & 0x0F) == (ori_int_tensor[row, unpacked_col] & 0x0F)

            # Upper 4 bits should be the second value
            assert ((pack_tensor[row, col] >> 4) & 0x0F) == (ori_int_tensor[row, unpacked_col + 1] & 0x0F)

    def test_pack_on_row_fast_248bit_2bit(self):
        # Setup
        ori_int_tensor = torch.randint(0, 4, (128, 64), dtype=torch.int)
        pack_tensor = torch.zeros((128, 16), dtype=torch.uint8)

        # Execute
        HQQWeightOnlyQuantizer.pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits=2)

        # Assert
        assert pack_tensor.shape == (128, 16)  # 4 values per byte for 2-bit

    def test_pack_on_row_fast_248bit_8bit(self):
        # Setup
        ori_int_tensor = torch.randint(0, 256, (128, 64), dtype=torch.int)
        pack_tensor = torch.zeros((128, 64), dtype=torch.uint8)

        # Execute
        HQQWeightOnlyQuantizer.pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits=8)

        # Assert
        # For 8-bit, packing is essentially a copy
        assert torch.all(pack_tensor == (ori_int_tensor & 0xFF))

    def test_pack_on_row_fast_248bit_invalid_bits(self):
        # Setup
        ori_int_tensor = torch.randint(0, 16, (128, 64), dtype=torch.int)
        pack_tensor = torch.zeros((128, 32), dtype=torch.uint8)

        # Execute
        with pytest.raises(NotImplementedError):
            HQQWeightOnlyQuantizer.pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits=3)

    @patch("torch.from_numpy")
    @patch("torch.cuda.is_available", return_value=False)
    def test_quantize_internal(self, mock_cuda_available, mock_from_numpy, default_config):
        # Setup
        mock_tensor = torch.randn(256, 128)
        mock_from_numpy.return_value = mock_tensor
        quantizer = HQQWeightOnlyQuantizer(default_config)

        # Execute
        w_q, scale, zero_point = quantizer.quantize_internal(
            mock_tensor, bits=4, channel_wise=True, group_size=128, optimize=True, round_zero=True, axis=1
        )

        # Assert
        assert w_q.shape == mock_tensor.shape
        assert scale.dtype == mock_tensor.dtype
        assert zero_point.dtype == mock_tensor.dtype

    def test_quantize_gather_node(self, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = "Gather"
        node.name = "test_gather"
        graph_stack = [MagicMock(spec=GraphProto)]

        # Execute
        quantizer = HQQWeightOnlyQuantizer(default_config)
        with pytest.raises(NotImplementedError):
            quantizer.quantize(node, graph_stack)

    @patch("olive.passes.onnx.matmul_quant.hqq_quantizer.HQQWeightOnlyQuantizer.quantize_internal")
    @patch("olive.passes.onnx.matmul_quant.hqq_quantizer.get_initializer")
    def test_quantize_matmul_node(self, mock_get_initializer, mock_quantize_internal, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = "MatMul"
        node.name = "test_matmul"
        node.input = ["input_a", "input_b"]
        node.output = ["output"]

        data = np.random.rand(256, 128).astype(np.float32)
        tensor_proto = onnx.helper.make_tensor(
            name="input_b", data_type=TensorProto.FLOAT, dims=data.shape, vals=data.flatten().tolist()
        )

        graph_proto = MagicMock(spec=GraphProto)
        graph_proto.input = [MagicMock()]
        graph_proto.input[0].name = "input_b"

        mock_get_initializer.return_value = (tensor_proto, graph_proto)

        # Setup quantize_internal return values
        quant_weight = torch.randint(0, 16, (128, 256), dtype=torch.int)
        scales = torch.rand(128, 1)
        zero_points = torch.rand(128, 1)
        mock_quantize_internal.return_value = (quant_weight, scales, zero_points)

        # Execute
        quantizer = HQQWeightOnlyQuantizer(default_config)
        with patch.object(quantizer, "pack_on_row_fast_248bit") as mock_pack:
            result = quantizer.quantize(node, [graph_proto])

        # Assert
        assert len(result) == 1
        assert result[0].op_type == str(OpType.MatMulNBits)
        assert result[0].domain == MSFT_DOMAIN

        # Verify the node has the expected attributes
        assert len(result[0].attribute) >= 4
        attribute_names = [attr.name for attr in result[0].attribute]
        assert "K" in attribute_names
        assert "N" in attribute_names
        assert "bits" in attribute_names
        assert "block_size" in attribute_names

        # Verify that initializers were added to the graph
        assert len(graph_proto.initializer.extend.call_args_list) > 0
        assert mock_pack.called

    @patch("olive.passes.onnx.matmul_quant.hqq_quantizer.get_initializer")
    def test_quantize_matmul_non_2d(self, mock_get_initializer, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = "MatMul"
        node.name = "test_matmul"
        node.input = ["input_a", "input_b"]

        tensor_proto = MagicMock(spec=TensorProto)
        tensor_proto.name = "input_b"

        graph_proto = MagicMock(spec=GraphProto)

        mock_get_initializer.return_value = (tensor_proto, graph_proto)

        # Set up a 3D array to trigger the non-2D path
        with patch("onnx.numpy_helper.to_array", return_value=np.random.rand(2, 3, 4)):
            # Execute
            quantizer = HQQWeightOnlyQuantizer(default_config)
            result = quantizer.quantize(node, [graph_proto])

        # Assert
        # Should return original node since weight is not 2D
        assert len(result) == 1
        assert result[0] is node

    @patch("olive.passes.onnx.matmul_quant.hqq_quantizer.get_initializer")
    def test_quantize_matmul_no_const_weight(self, mock_get_initializer, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = "MatMul"
        node.name = "test_matmul"
        node.input = ["input_a", "input_b"]

        # Return None for the tensor to simulate no constant weight
        mock_get_initializer.return_value = (None, None)

        # Execute
        quantizer = HQQWeightOnlyQuantizer(default_config)
        result = quantizer.quantize(node, [MagicMock()])

        # Assert
        # Should return original node since there's no constant weight
        assert len(result) == 1
        assert result[0] is node
