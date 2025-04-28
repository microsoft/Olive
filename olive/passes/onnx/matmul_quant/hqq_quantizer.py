import logging

import onnx
import torch
from onnx import GraphProto, NodeProto

from olive.passes.onnx.matmul_quant.utils import MSFT_DOMAIN, Algorithm, OpType, WeightOnlyQuantConfig, get_initializer

logger = logging.getLogger(__name__)


class HQQWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        block_size=128,
        bits=4,
        axis=1,
        op_types_to_quantize: tuple[str, ...] | None = None,
        quant_axes: tuple[tuple[str, int], ...] | None = None,
    ):
        """Configure HQQ Weight Only Quantization parameters.

        HQQ algorithm quant weight without needing calibrate data.

        Args:
            block_size (int, optional):
                channel number in one block to execute a HQQ quantization iteration.
            bits (int, optional):
                how many bits to represent weight.
            axis (int, optional):
                0 or 1. which axis to quantize. https://arxiv.org/pdf/2309.15531.pdf
            op_types_to_quantize (optional):
                set of operator types to quantize.
            quant_axes (dict[str, int], optional):
                op:axis, which axis to quantize for an op. Default {MatMul: 0, Gather: 1}

        """
        from onnxruntime.quantization import QuantFormat

        super().__init__(
            algorithm=Algorithm.HQQ,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=op_types_to_quantize,
            quant_axes=quant_axes,
        )
        self.block_size = block_size
        self.bits = bits
        self.axis = axis


class HQQWeightOnlyQuantizer:
    """HQQ Quantizer class for quantizing ONNX models.

    HQQ only supports QOperator format
    """

    def __init__(
        self,
        config: HQQWeightOnlyQuantConfig,
    ):
        self.config = config

    # Proximal solver || weight - dequantize(quantize(weight))||_p^p
    @staticmethod
    def optimize_weights(tensor, scale, zero, min_max: list[int], axis: int = 0):
        lp_norm = 0.7
        beta = 1e1
        kappa = 1.01
        iters = 20

        dtype = torch.float16 if tensor.is_cuda else torch.float32
        w_f = tensor.to(dtype)
        scale = scale.to(dtype)
        zero = zero.to(dtype)

        def shrink_op(x, beta, p=lp_norm):
            if p == 1:
                # Formula: sign(x) * max(abs(x) - 1/beta, 0)
                return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
            else:
                # Formula: sign(x) * max(abs(x) - (1/beta) * abs(x)^(p-1), 0)
                return torch.sign(x) * torch.nn.functional.relu(
                    torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x) + 1e-8, p - 1)
                )

        best_error = 1e4
        for _ in range(iters):
            w_q = torch.round(w_f * scale + zero).clamp(min_max[0], min_max[1])
            w_r = (w_q - zero) / scale
            w_e = shrink_op(w_f - w_r, beta)
            zero = torch.mean(w_q - (w_f - w_e) * scale, axis=axis, keepdim=True)
            beta *= kappa

            current_error = float(torch.abs(w_f - w_r).mean())
            if current_error < best_error:
                best_error = current_error
            else:
                break

        del w_f, w_q, w_r, w_e

        return scale, zero

    @staticmethod
    def pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
        if pack_tensor.shape[0] == ori_int_tensor.shape[0]:
            ori_int_tensor = ori_int_tensor.T
            pack_tensor = pack_tensor.T
        if bits in [2, 4, 8]:
            compress_ratio = pack_tensor.element_size() * 8 // bits
            for j in range(compress_ratio):
                pack_tensor[0:] |= ori_int_tensor[j::compress_ratio] << (bits * (j))
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

    # from Official implementation of Half-Quadratic Quantization (HQQ)
    def quantize_internal(
        self, tensor, bits=4, channel_wise=True, group_size=64, optimize=True, round_zero=True, axis=1
    ):
        weight = tensor.float()
        ori_shape = weight.shape

        pad_len = (group_size - ori_shape[axis] % group_size) % group_size
        if axis == 1:
            weight = torch.nn.functional.pad(weight, (0, pad_len), "constant", 0)
        else:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_len), "constant", 0)
        shape = weight.shape

        # Reshape for grouping
        if (group_size is not None) and channel_wise:
            weight = weight.reshape([-1, group_size]) if (axis == 1) else weight.reshape([group_size, -1])

        # Get min/max values
        if channel_wise is False:
            _min, _max = weight.min(), weight.max()
            optimize = False
        else:
            _min = weight.min(axis=axis, keepdim=True)[0]
            _max = weight.max(axis=axis, keepdim=True)[0]

        max_v = 2**bits - 1
        min_v = 0
        min_max = [min_v, max_v]

        # Note: here we work with the inverse of the scale to avoid division and quantize instead via weight*scale + zero, the scale is inverted later on.
        # clamp to avoid half-precision problems
        scale = (max_v / (_max - _min)).clamp(max=2e4)
        min_max_axis = _max - _min
        if (min_max_axis == 0).sum().item() > 0:
            min_max_axis[min_max_axis == 0] = max_v
            scale = (max_v / min_max_axis).clamp(max=2e4)
        zero = -_min * scale

        if round_zero:
            zero = torch.round(zero)

        # Fine-tune weights
        if optimize:
            scale, zero = self.optimize_weights(tensor=weight, scale=scale, zero=zero, min_max=min_max, axis=axis)

        # Quantize
        # Necessary for fake quantization backprop
        w_q = torch.round(weight * scale + zero).clamp(min_max[0], min_max[1])
        w_q = w_q.reshape(shape).int()

        scale = 1.0 / scale
        if axis == 1:
            scale = scale.reshape(shape[0], -1)
            zero = zero.reshape(shape[0], -1)
        else:
            scale = scale.reshape(-1, shape[-1])
            zero = zero.reshape(-1, shape[-1])
        # cleanup
        del weight, _min, _max

        return w_q, scale.to(tensor.dtype), zero.to(tensor.dtype)

    def quantize(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]:
        """Quantize the weight of the target node and return the new nodes.

        Target node:        QOperator node:
        MatMul              MatMulNBits
        If the node is target node with fp32 or fp16 const weight, quantize the weight to int4 and
        return the new nodes.
        Return the corresponding QOperator nodes.
        """
        # With HQQ, zero points are in float. Current GatherBlockQuantized does not support float zero points.
        if node.op_type == "Gather":
            raise NotImplementedError("Gather quantization is not supported yet in HQQ")

        logger.info("start to quantize %s ...", node.name)
        input_b = node.input[1]
        b_pb, bs_graph = get_initializer(input_b, graph_stack)
        if b_pb is None:
            logger.info("MatMul doesn't have const weight. Skip to quantize")
            return [node]  # only care about constant weight

        b_array = onnx.numpy_helper.to_array(b_pb)
        if len(b_array.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return [node]  # can only process 2-D matrix
        b_array_torch = torch.from_numpy(b_array)
        if torch.cuda.is_available():
            b_array_torch = b_array_torch.cuda()
        quant_weight_torch, scales_torch, zero_points_torch = self.quantize_internal(
            b_array_torch.T, bits=self.config.bits, group_size=self.config.block_size
        )
        quant_weight_torch = quant_weight_torch.contiguous()
        scales_torch = scales_torch.contiguous()
        zero_points_torch = zero_points_torch.contiguous()

        packed_torch = torch.zeros(
            (quant_weight_torch.shape[0], quant_weight_torch.shape[1] // 2),
            dtype=torch.uint8,
            device=quant_weight_torch.device,
        )
        self.pack_on_row_fast_248bit(packed_torch, quant_weight_torch, self.config.bits)
        scales = scales_torch.cpu().numpy()
        zero_points = zero_points_torch.cpu().numpy()
        # reshape to the predefined shape in MatmulNbits
        scales = scales.reshape(-1)
        zero_points = zero_points.reshape(-1)
        rows, cols = b_array_torch.shape
        block_size = self.config.block_size
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        packed_torch = packed_torch.reshape(cols, k_blocks, blob_size)

        b_quant = onnx.numpy_helper.from_array(packed_torch.cpu().numpy())
        b_quant.name = b_pb.name + "_Q4"
        for graph_input in bs_graph.input:
            if graph_input.name == input_b:
                bs_graph.input.remove(graph_input)
                break

        scales_tensor = onnx.numpy_helper.from_array(scales)
        scales_tensor.name = b_pb.name + "_scales"
        bs_graph.initializer.extend([b_quant, scales_tensor])

        input_names = [node.input[0], b_quant.name, scales_tensor.name]
        zp_tensor = onnx.numpy_helper.from_array(zero_points)
        zp_tensor.name = b_pb.name + "_zero_points"
        bs_graph.initializer.extend([zp_tensor])
        input_names.append(zp_tensor.name)

        kwargs = {}
        rows, cols = b_array.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = self.config.bits
        kwargs["block_size"] = self.config.block_size

        matmul_q4_node = onnx.helper.make_node(
            str(OpType.MatMulNBits),
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain=MSFT_DOMAIN,
            **kwargs,
        )

        logger.info("complete quantization of %s ...", node.name)

        return [matmul_q4_node]
