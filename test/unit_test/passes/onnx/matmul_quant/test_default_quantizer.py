# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from unittest import skipIf
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx import GraphProto, NodeProto, TensorProto
from onnxruntime import __version__ as OrtVersion
from packaging import version

from olive.passes.onnx.matmul_quant.default_quantizer import DefaultWeightOnlyQuantConfig, DefaultWeightOnlyQuantizer
from olive.passes.onnx.matmul_quant.utils import Algorithm, OpType

# pylint: disable=redefined-outer-name


@pytest.fixture
def default_config():
    return DefaultWeightOnlyQuantConfig(
        bits=4, block_size=128, is_symmetric=False, accuracy_level=None, op_types_to_quantize=None, quant_axes=None
    )


@pytest.fixture
def symmetric_config():
    return DefaultWeightOnlyQuantConfig(
        bits=4, block_size=128, is_symmetric=True, accuracy_level=None, op_types_to_quantize=None, quant_axes=None
    )


@pytest.fixture
def bits8_config():
    return DefaultWeightOnlyQuantConfig(
        bits=8, block_size=128, is_symmetric=False, accuracy_level=None, op_types_to_quantize=None, quant_axes=None
    )


class TestDefaultWeightOnlyQuantConfig:
    def test_init_default_values(self):
        config = DefaultWeightOnlyQuantConfig()
        assert config.block_size == 128
        assert config.is_symmetric is False
        assert config.bits == 4
        assert config.accuracy_level is None
        assert config.algorithm == Algorithm.DEFAULT

    def test_init_custom_values(self):
        config = DefaultWeightOnlyQuantConfig(
            bits=8,
            block_size=64,
            is_symmetric=True,
            accuracy_level=2,
            op_types_to_quantize=("MatMul",),
            quant_axes=(("MatMul", 0),),
        )
        assert config.block_size == 64
        assert config.is_symmetric is True
        assert config.bits == 8
        assert config.accuracy_level == 2
        assert config.algorithm == Algorithm.DEFAULT
        assert config.op_types_to_quantize == {"MatMul"}
        assert config.quant_axes == {"MatMul": 0}


class TestDefaultWeightOnlyQuantizer:
    def test_init(self, default_config):
        quantizer = DefaultWeightOnlyQuantizer(default_config)
        assert quantizer.config == default_config

    def _assert_quantize_matmul_call_args(
        self, mock_quantize, packed, fp32weight, scales, zero_point, block_size, cols, rows, is_symmetric
    ):
        """Helper function to assert quantize_matmul_4bits call arguments"""
        assert mock_quantize.called
        call_args = mock_quantize.call_args[0]

        assert call_args[0] is packed
        assert call_args[1].shape[0] >= fp32weight.shape[0]
        assert call_args[1].shape[1] == fp32weight.shape[1]
        assert call_args[2] is scales
        assert call_args[3] is zero_point
        assert call_args[4] == block_size
        assert call_args[5] == cols
        assert call_args[6] >= rows
        assert call_args[7] is is_symmetric

    @patch("onnxruntime.capi._pybind_state.quantize_matmul_4bits")
    def test_qbits_block_quant_4bits_asymmetric(self, mock_quantize, default_config):
        # Setup
        rows = 256
        cols = 128
        fp32weight = np.random.rand(rows, cols).astype(np.float32)

        quantizer = DefaultWeightOnlyQuantizer(default_config)

        # Execute
        packed, scales, zero_point = quantizer.qbits_block_quant(fp32weight)

        # Assert
        self._assert_quantize_matmul_call_args(
            mock_quantize=mock_quantize,
            packed=packed,
            fp32weight=fp32weight,
            scales=scales,
            zero_point=zero_point,
            block_size=default_config.block_size,
            cols=cols,
            rows=rows,
            is_symmetric=default_config.is_symmetric,
        )

    @patch("onnxruntime.capi._pybind_state.quantize_matmul_4bits")
    def test_qbits_block_quant_4bits_symmetric(self, mock_quantize, symmetric_config):
        # Setup
        rows = 256
        cols = 128
        fp32weight = np.random.rand(rows, cols).astype(np.float32)

        quantizer = DefaultWeightOnlyQuantizer(symmetric_config)

        # Execute
        packed, scales, zero_point = quantizer.qbits_block_quant(fp32weight)

        # Assert
        self._assert_quantize_matmul_call_args(
            mock_quantize=mock_quantize,
            packed=packed,
            fp32weight=fp32weight,
            scales=scales,
            zero_point=zero_point,
            block_size=symmetric_config.block_size,
            cols=cols,
            rows=rows,
            is_symmetric=symmetric_config.is_symmetric,
        )

    @skipIf(
        version.parse(OrtVersion) <= version.parse("1.21.1"),
        reason="Default quantization of 8-bit weights is not supported for onnxruntime<=1.21.1",
    )
    @patch("onnxruntime.capi._pybind_state.quantize_matmul_8bits")
    def test_qbits_block_quant_8bits(self, mock_quantize, bits8_config):
        # Setup
        rows = 256
        cols = 128
        fp32weight = np.random.rand(rows, cols).astype(np.float32)

        quantizer = DefaultWeightOnlyQuantizer(bits8_config)

        # Execute
        packed, scales, zero_point = quantizer.qbits_block_quant(fp32weight)

        # Assert
        self._assert_quantize_matmul_call_args(
            mock_quantize=mock_quantize,
            packed=packed,
            fp32weight=fp32weight,
            scales=scales,
            zero_point=zero_point,
            block_size=bits8_config.block_size,
            cols=cols,
            rows=rows,
            is_symmetric=bits8_config.is_symmetric,
        )

    def test_quant_slice_symmetric(self):
        # Setup
        data = np.array([[1.0, 2.0, -3.0, 4.0], [-5.0, 6.0, 7.0, -8.0]])

        # Execute
        quantized_slice, scale = DefaultWeightOnlyQuantizer.quant_slice_symmetric(data)

        # Assert
        expected_scale = np.array([[-0.5], [1.0]], dtype=np.float32)
        expected_quantized_slice = np.array([[-2, -4, 6, -8], [-5, 6, 7, -8]], dtype=np.int8)

        np.testing.assert_array_equal(quantized_slice, expected_quantized_slice)
        np.testing.assert_array_equal(scale, expected_scale)

    def test_quant_slice_asymmetric(self):
        # Setup
        data = np.array([[1.0, 2.0, -3.0, 4.0], [-5.0, 6.0, 7.0, -8.0]])

        # Execute
        quantized_slice, scale, zero_point = DefaultWeightOnlyQuantizer.quant_slice_asymmetric(data)

        # Assert
        expected_quantized_slice = np.array([[8, 10, 0, 15], [3, 14, 15, 0]], dtype=np.uint8)
        expected_scale = np.array([[0.4666667], [1.0]], dtype=np.float32)
        expected_zero_point = np.array([[6], [8]], dtype=np.uint8)

        np.testing.assert_array_equal(quantized_slice, expected_quantized_slice)
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)
        np.testing.assert_array_equal(zero_point, expected_zero_point)

    def test_pack_int8_to_int4(self):
        # Setup
        data = np.array([1, -2, 3, -4, 5, -6, 7, -8], dtype=np.int8)

        # Execute
        packed = DefaultWeightOnlyQuantizer.pack_int8_to_int4(data)

        # Assert
        assert packed.shape == (4,)
        assert packed.dtype == np.uint8

        # Manual verification of packing (lower 4 bits from first value, upper 4 bits from second value)
        expected = np.array(
            [
                (1 & 0xF) | (((-2) & 0xF) << 4),
                (3 & 0xF) | (((-4) & 0xF) << 4),
                (5 & 0xF) | (((-6) & 0xF) << 4),
                (7 & 0xF) | (((-8) & 0xF) << 4),
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(packed, expected)

    @pytest.mark.parametrize(
        "is_symmetric, quantize_axis, block_size, data_shape, expected_scales_shape",
        [
            # Symmetric quantization, quantize along axis 0
            (True, 0, 8, (16, 32), (2, 32)),
            # Asymmetric quantization, quantize along axis 0
            (False, 0, 8, (16, 32), (2, 32)),
            # Symmetric quantization, quantize along axis 1
            (True, 1, 16, (16, 32), (16, 2)),
            # Asymmetric quantization, quantize along axis 1
            (False, 1, 16, (16, 32), (16, 2)),
            # Using smaller block size
            (True, 0, 4, (16, 32), (4, 32)),
            # Using dimension not divisible by block size
            (False, 0, 10, (25, 32), (3, 32)),
        ],
    )
    def test_quantize_ndarray(self, is_symmetric, quantize_axis, block_size, data_shape, expected_scales_shape):
        # Setup
        data = np.random.rand(*data_shape).astype(np.float32)

        # Execute
        _, scales, zero_points = DefaultWeightOnlyQuantizer.quantize_ndarray(
            data, quantize_axis=quantize_axis, block_size=block_size, is_symmetric=is_symmetric
        )

        # Assert
        assert scales.shape == expected_scales_shape

        if is_symmetric:
            assert zero_points is None
        else:
            assert zero_points is not None

    @patch("olive.passes.onnx.matmul_quant.default_quantizer.DefaultWeightOnlyQuantizer.quantize_matmul")
    def test_quantize_matmul_node(self, mock_quantize_matmul, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = "MatMul"
        node.name = "test_matmul"
        graph_stack = [MagicMock(spec=GraphProto)]
        quantizer = DefaultWeightOnlyQuantizer(default_config)

        # Execute
        quantizer.quantize(node, graph_stack)

        # Assert
        mock_quantize_matmul.assert_called_with(node, graph_stack)

    @patch("olive.passes.onnx.matmul_quant.default_quantizer.DefaultWeightOnlyQuantizer.quantize_gather")
    def test_quantize_gather_node(self, mock_quantize_gather, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = "Gather"
        node.name = "test_gather"
        graph_stack = [MagicMock(spec=GraphProto)]
        quantizer = DefaultWeightOnlyQuantizer(default_config)

        # Execute
        quantizer.quantize(node, graph_stack)

        # Assert
        mock_quantize_gather.assert_called_with(node, graph_stack)

    def test_quantize_unsupported_node(self, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = "Conv"
        node.name = "test_conv"
        graph_stack = [MagicMock(spec=GraphProto)]
        quantizer = DefaultWeightOnlyQuantizer(default_config)

        # Execute
        result = quantizer.quantize(node, graph_stack)

        # Assert
        assert result == [node]

    @patch("olive.passes.onnx.matmul_quant.default_quantizer.get_initializer")
    @patch("olive.passes.onnx.matmul_quant.default_quantizer.DefaultWeightOnlyQuantizer.quantize_ndarray")
    def test_quantize_gather(self, mock_quantize_ndarray, mock_get_initializer, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = str(OpType.Gather)
        node.name = "test_gather"
        node.input = ["data", "indices"]
        node.output = ["output"]

        tensor_proto = MagicMock(spec=TensorProto)
        tensor_proto.name = "data"

        graph_proto = MagicMock(spec=GraphProto)
        graph_proto.input = [MagicMock()]
        graph_proto.input[0].name = "data"
        graph_proto.initializer = []

        mock_get_initializer.return_value = (tensor_proto, graph_proto)

        data_ndarray = np.random.rand(128, 256).astype(np.float32)

        # Mock quantized results
        quantized_data = np.zeros(16384, dtype=np.uint8)
        scales = np.zeros((8, 256), dtype=np.float32)
        zero_points = np.zeros((8, 128), dtype=np.uint8)
        mock_to_array = MagicMock(return_value=data_ndarray)
        mock_quantize_ndarray.return_value = (quantized_data, scales, zero_points)

        quantizer = DefaultWeightOnlyQuantizer(default_config)

        with patch("onnx.numpy_helper.to_array", mock_to_array):
            # Execute
            result = quantizer.quantize_gather(node, [graph_proto])

        # Assert
        assert mock_quantize_ndarray.called
        assert len(result) == 1
        assert result[0].op_type == str(OpType.GatherBlockQuantized)
        assert len(graph_proto.initializer) > 0  # Tensors should be added to graph initializers

    @patch("olive.passes.onnx.matmul_quant.default_quantizer.get_initializer")
    def test_quantize_gather_no_const_weight(self, mock_get_initializer, default_config):
        # Setup
        node = MagicMock(spec=NodeProto)
        node.op_type = str(OpType.Gather)
        node.name = "test_gather"
        node.input = ["data", "indices"]
        mock_get_initializer.return_value = (None, None)

        quantizer = DefaultWeightOnlyQuantizer(default_config)

        # Execute
        result = quantizer.quantize_gather(node, [MagicMock()])

        # Assert
        assert result == [node]
