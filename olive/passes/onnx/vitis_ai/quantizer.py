#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import logging

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer, QDQQuantTensorType, QDQTensorQuantInfo
from onnxruntime.quantization.quant_utils import (
    DEQUANT_OP_NAME,
    QUANT_OP_NAME,
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizedValue,
    QuantizedValueType,
    QuantType,
    __producer__,
    __version__,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_input_suffix,
    add_quant_output_suffix,
    add_quant_suffix,
    find_by_name,
    get_qmin_qmax_for_qType,
    tensor_proto_to_array,
)
from onnxruntime.quantization.registry import CreateQDQQuantizer

from olive.passes.onnx.vitis_ai.quant_utils import (
    PowerOfTwoMethod,
    compute_scale_zp_pof2s,
    convert_relu_input_to_annotate_output,
    get_annotate_output_name,
    get_qdq_to_remove,
    get_relu_name,
    quantize_data_pof2s,
    remove_nodes,
    vitis_quantize_data,
)
from olive.passes.onnx.vitis_ai.refine import adjust_quantize_info

logger = logging.getLogger(__name__)

# ruff: noqa


class VitisDPUQuantizer(QDQQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        calibrate_method,
        need_layer_fusing,
        extra_options=None,
    ):
        QDQQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.calibrate_method = calibrate_method
        self.need_layer_fusing = need_layer_fusing

        if per_channel:
            logger.warning(
                "Only per-tensor quantization is supported when enable_dpu=True, `per_channel` will be set to False."
            )

        if activation_qType != QuantType.QInt8:
            logger.warning(
                "Only QuantType.QInt8 activation_type is supported is supported when enable_dpu=True, "
                "`activation_type` will be set to QuantType.QInt8."
            )

        if weight_qType != QuantType.QInt8:
            logger.warning(
                "Only QuantType.QInt8 weight_type is supported when enable_dpu=True, `weight_type` will "
                "be set to QuantType.QInt8."
            )

        # If using enable_dpu, QDQ should always appear as a pair.
        # Therefore, we need to add qdq pair to weight.
        if "AddQDQPairToWeight" in self.extra_options and not self.extra_options["AddQDQPairToWeight"]:
            logger.warning("When using enable_dpu, AddQDQPairToWeight will be changed to true.")
        self.add_qdq_pair_to_weight = True

        # If using nable_dpu, QDQ should always set WeightSymmetric as True.
        if "WeightSymmetric" in self.extra_options and not self.extra_options["WeightSymmetric"]:
            logger.warning("When enable_dpu=True, WeightSymmetric will be set to true.")
        self.is_weight_symmetric = True

        # If using enable_dpu, QDQ should always always set ActivationSymmetric as True.
        if "ActivationSymmetric" in self.extra_options and not self.extra_options["ActivationSymmetric"]:
            logger.warning("When enable_dpu=True, ActivationSymmetric will be set to true.")
        self.is_activation_symmetric = True

    def vitis_quantize_initializer(self, weight, bit_width=8, keep_float_weight=False):

        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name = weight.name + "_quantized"
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Update packed weight, zero point, and scale initializers
        weight_data = tensor_proto_to_array(weight)
        _, _, zero_point, scale, q_weight_data = vitis_quantize_data(
            weight_data.flatten(), bit_width, method=self.calibrate_method
        )
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, [], [scale])
        zero_initializer = onnx.helper.make_tensor(zp_name, onnx_proto.TensorProto.INT8, [], [zero_point])
        self.model.initializer().extend([scale_initializer, zero_initializer])

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return q_weight_name, zp_name, scale_name

    def quantize_model(self):

        self.tensor_info = {}
        model = self.model.model
        annotate_output_name_list = get_annotate_output_name(model)
        relu_to_conv_output = get_relu_name(model, annotate_output_name_list)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()

        self._quantize_sharing_param_tensors()
        dq_nodes_to_remove, q_nodes_to_remove = get_qdq_to_remove(model, relu_to_conv_output)
        convert_relu_input_to_annotate_output(model, relu_to_conv_output)
        if self.need_layer_fusing:
            model = remove_nodes(model, dq_nodes_to_remove)
            model = remove_nodes(model, q_nodes_to_remove)
        self._quantize_refine()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        model.producer_name = __producer__
        model.producer_version = __version__

        return model

    def _add_qdq_pair_for_initializer(self, weight_proto, tensor_type, axis=None):
        weight_name = weight_proto.name
        q_weight_name, zp_name, scale_name = self.vitis_quantize_initializer(
            weight_proto, self.weight_qType, keep_float_weight=True
        )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )

            self.model.add_node(dequant_node)

    def quantize_bias_tensor(self, bias_name, input_name, weight_name, beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                # Use int8 quantization for bias as well as weights.
                self.tensors_to_quantize[bias_name] = QDQTensorQuantInfo()
        else:
            logger.warning("Expected {} to be a weight".format(bias_name))

    def _quantize_refine(self):
        self.model = adjust_quantize_info(
            self.model,
            adjust_vitis_sigmoid=True,
            adjust_shift_cut=True,
            adjust_shift_bias=True,
            adjust_shift_read=True,
            adjust_shift_write=True,
            align_concat=True,
            align_pool=True,
        )


class VitisQOpQuantizer(ONNXQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        calibrate_method,
        extra_options=None,
    ):
        ONNXQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options=None,
        )
        self.calibrate_method = calibrate_method

    def quantize_initializer(self, weight, qType, reduce_range=False, keep_float_weight=False):
        """
        :param weight: TensorProto initializer
        :param qType: type to quantize to
        :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to
                                  qunatize scale and zero point. If keep_float_weight is False,
                                  quantize the weight, or don't quantize the weight.
        :return: quantized weight name, zero point name, scale name
        """
        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name = weight.name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Update packed weight, zero point, and scale initializers
        weight_data = tensor_proto_to_array(weight)
        _, _, zero_point, scale, q_weight_data = quantize_data_pof2s(
            weight_data.flatten().tolist(),
            qType,
            self.is_weight_symmetric,
            self.reduce_range and reduce_range,
            method=PowerOfTwoMethod.NonOverflow,
        )
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, [], [scale])
        zero_initializer = onnx.helper.make_tensor(zp_name, qType, [], [zero_point])
        self.model.initializer().extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            q_weight_data = np.asarray(q_weight_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[qType]).reshape(
                weight.dims
            )
            q_weight_initializer = onnx.numpy_helper.from_array(q_weight_data, q_weight_name)
            self.model.initializer().extend([q_weight_initializer])

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return q_weight_name, zp_name, scale_name

    def quantize_weight_per_channel(
        self,
        weight_name,
        weight_qType,
        channel_axis,
        reduce_range=True,
        keep_float_weight=False,
    ):
        # Find if this input is already quantized
        if weight_name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight_name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        initializer = find_by_name(weight_name, self.model.initializer())
        if initializer is None:
            raise ValueError("{} is not an initializer", weight_name)

        weights = tensor_proto_to_array(initializer)
        channel_count = weights.shape[channel_axis]
        rmin_list = []
        rmax_list = []
        zero_point_list = []
        scale_list = []
        quantized_per_channel_data_list = []
        for i in range(channel_count):
            per_channel_data = weights.take(i, channel_axis)
            rmin, rmax, zero_point, scale, quantized_per_channel_data = quantize_data_pof2s(
                per_channel_data.flatten().tolist(),
                weight_qType,
                self.is_weight_symmetric or weight_qType == onnx_proto.TensorProto.INT8,
                self.reduce_range and reduce_range,
                method=PowerOfTwoMethod.NonOverflow,
            )
            rmin_list.append(rmin)
            rmax_list.append(rmax)
            zero_point_list.append(zero_point)
            scale_list.append(scale)
            quantized_per_channel_data_list.append(quantized_per_channel_data)

        # combine per_channel_data into one
        reshape_dims = list(weights.shape)  # deep copy
        reshape_dims[channel_axis] = 1  # only one per channel for reshape
        quantized_weights = np.asarray(quantized_per_channel_data_list[0]).reshape(reshape_dims)
        for i in range(1, len(quantized_per_channel_data_list)):
            channel_weights = np.asarray(quantized_per_channel_data_list[i]).reshape(reshape_dims)
            quantized_weights = np.concatenate((quantized_weights, channel_weights), channel_axis)

        q_weight_name = weight_name + TENSOR_NAME_QUANT_SUFFIX
        zp_name = weight_name + "_zero_point"
        scale_name = weight_name + "_scale"

        quantized_value = QuantizedValue(
            weight_name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight_name] = quantized_value

        # Update packed weight, zero point, and scale initializers
        zero_scale_shape = [initializer.dims[channel_axis]]
        scale_initializer = onnx.helper.make_tensor(
            scale_name, onnx_proto.TensorProto.FLOAT, zero_scale_shape, scale_list
        )
        zero_initializer = onnx.helper.make_tensor(zp_name, weight_qType, zero_scale_shape, zero_point_list)

        self.model.initializer().extend([scale_initializer, zero_initializer])

        if not keep_float_weight:
            quantized_weights = np.asarray(
                quantized_weights,
                dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[weight_qType],
            ).reshape(initializer.dims)
            q_weight_initializer = onnx.numpy_helper.from_array(quantized_weights, q_weight_name)
            self.model.initializer().extend([q_weight_initializer])

        return q_weight_name, zp_name, scale_name

    def calculate_quantization_params(self):
        from olive.passes.onnx.vitis_ai.quant_utils import is_ort_version_below_1_16

        if self.tensors_range is None:
            return

        # adjust tensor_ranges for input of Clip and Relu node
        for node in self.model.nodes():
            if node.op_type not in ["Clip", "Relu"]:
                continue
            if self.is_activation_symmetric:
                continue
            if not self.should_quantize_node(node):
                continue
            if len(self.model.input_name_to_nodes()[node.input[0]]) != 1:
                continue
            if node.input[0] not in self.tensors_range or node.output[0] not in self.tensors_range:
                continue
            self.tensors_range[node.input[0]] = self.tensors_range[node.output[0]]
        quantization_params = {}
        if is_ort_version_below_1_16():
            for tensor_name in self.tensors_range.keys():
                rmin, rmax = self.tensors_range[tensor_name]
                qmin, qmax = get_qmin_qmax_for_qType(self.activation_qType, symmetric=self.is_activation_symmetric)

                quantization_params[tensor_name] = compute_scale_zp_pof2s(
                    rmin, rmax, qmin, qmax, self.is_activation_symmetric
                )
        else:
            from onnxruntime.quantization.onnx_quantizer import QuantizationParams

            for tensor_name in self.tensors_range:
                td = self.tensors_range[tensor_name]
                rmin, rmax = td.range_value
                qmin, qmax = get_qmin_qmax_for_qType(self.activation_qType, symmetric=self.is_activation_symmetric)

                zero, scale = compute_scale_zp_pof2s(rmin, rmax, qmin, qmax, self.is_activation_symmetric)
                quantization_params[tensor_name] = QuantizationParams(zero_point=zero, scale=scale)

        return quantization_params


class VitisQDQQuantizer(VitisQOpQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):
        ONNXQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.bias_to_quantize = []

        self.nodes_to_remove = []

        # Specific op types to exclude qdq quantization for their outputs.
        # In TRT, it's not recommended to quantize outputs for weighted ops such as Conv, Matmul, Gemm
        # because those ops may be followed by nodes that require high resolution inputs.
        # Adding QDQ for those ops' output may end up with worse accuracy.
        # So, we don't recommend to add QDQ to node's output under such condition.
        self.op_types_to_exclude_output_quantization = (
            []
            if "OpTypesToExcludeOutputQuantization" not in extra_options
            else extra_options["OpTypesToExcludeOutputQuantization"]
        )

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = (
            False if "AddQDQPairToWeight" not in extra_options else extra_options["AddQDQPairToWeight"]
        )

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can’t be shared between nodes, so it will create dedicated QDQ pairs for each node.
        self.dedicated_qdq_pair = (
            False if "DedicatedQDQPair" not in extra_options else extra_options["DedicatedQDQPair"]
        )
        if self.dedicated_qdq_pair:
            self.tensor_to_its_receiving_nodes = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization
        # is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = (
            {}
            if "QDQOpTypePerChannelSupportToAxis" not in extra_options
            else extra_options["QDQOpTypePerChannelSupportToAxis"]
        )

    def _is_tensor_quantizable(self, tensor_name):
        """
        Check if tensor can be quantized
        """
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                return True
        elif tensor_name in self.value_infos.keys():
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                return True
        else:
            logger.warning(
                "failed to infer the type of tensor: {}. Skip to quantize it. Please check if it is expected.".format(
                    tensor_name
                )
            )

        return False

    def __quantize_tensor(self, tensor_name, quant_sharing_param=None, tensor_type=QDQQuantTensorType.ACTIVATION):
        """
        Quantize tensors. If quant_param_tensor is not None, tensor with name tensor_name will be quantized with same
        quantization parameters as tensor quant_param_tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter
            tensor_type: QDQQuantTensorType default ACTIVATION
        """
        if self._is_tensor_quantizable(tensor_name):
            if quant_sharing_param:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=tensor_type, quant_para_provider=quant_sharing_param
                )
            elif tensor_name not in self.tensors_to_quantize:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(tensor_type=tensor_type)

    def quantize_activation_tensor(self, tensor_name, quant_sharing_param=None):
        """
        Quantize Activation Tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter
        """
        return self.__quantize_tensor(tensor_name, quant_sharing_param, QDQQuantTensorType.ACTIVATION)

    def quantize_weight_tensor(self, tensor_name, quant_sharing_param=None):
        """
        Quantize Weight Tensor
        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_param: name of the tensor that provides quantization parameter
        """
        return self.__quantize_tensor(tensor_name, quant_sharing_param, QDQQuantTensorType.WEIGHT)

    def quantize_weight_tensor_per_channel(self, tensor_name, axis):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=QDQQuantTensorType.WEIGHT, axis=axis
                )
        else:
            logger.warning(
                "only support per-channel quantization on weight. Tensor: {} is not quantized.".format(tensor_name)
            )

    def quantize_bias_tensor(self, bias_name, input_name, weight_name, beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                self.bias_to_quantize.append((bias_name, input_name, weight_name, beta))
        else:
            logger.warning("Expected {} to be a weight".format(bias_name))

    def remove_node(self, node):
        self.nodes_to_remove.append(node)

    def remove_nodes(self):
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        self._quantize_bias_tensors()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if (
            output_name in self.quantization_params.keys()
            and len(self.model.input_name_to_nodes()[upstream_output_name]) == 1
            and not self.model.is_graph_output(upstream_output_name)
            and not self.model.is_graph_input(upstream_output_name)
        ):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    def _create_qdq_nodes(
        self, q_input, q_output, quant_node_name, dq_input, dq_output, dequant_node_name, scale_name, zp_name, axis=None
    ):
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_qdq_pair_for_initializer(self, weight_proto, tensor_type, axis=None):
        weight_name = weight_proto.name
        if axis is not None:
            if self.opset_version < 13:
                raise ValueError("Per-Channel support with QDQ format requires onnx opset version 13 or above.")
            q_weight_name, zp_name, scale_name = self.quantize_weight_per_channel(
                weight_name, onnx_proto.TensorProto.INT8, axis, keep_float_weight=self.add_qdq_pair_to_weight
            )
        else:
            q_weight_name, zp_name, scale_name = self.quantize_initializer(
                weight_proto,
                self.weight_qType if tensor_type is QDQQuantTensorType.WEIGHT else self.activation_qType,
                keep_float_weight=self.add_qdq_pair_to_weight,
            )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )
            self.model.add_node(dequant_node)

    def _add_qdq_pair_for_activation(self, tensor_name, scale_name, zp_name):
        if (
            self.dedicated_qdq_pair
            and tensor_name in self.tensor_to_its_receiving_nodes
            and len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1
        ):
            num_dedicated_qdq_pair = len(self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(tensor_name) + postfix
                quant_node_name_postfix = add_quant_suffix(tensor_name) + postfix
                dequant_node_name_postfix = add_dequant_suffix(tensor_name) + postfix
                self._create_qdq_nodes(
                    tensor_name,
                    tensor_name_quant_output_postfix,
                    quant_node_name_postfix,
                    tensor_name_quant_output_postfix,
                    tensor_name_dequant_output_postfix,
                    dequant_node_name_postfix,
                    scale_name,
                    zp_name,
                )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                    )
                    self.quantized_value_map[tensor_name] = quantized_value
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            self._create_qdq_nodes(
                q_input,
                add_quant_output_suffix(tensor_name),
                add_quant_suffix(tensor_name),
                add_quant_output_suffix(tensor_name),
                dq_output,
                add_dequant_suffix(tensor_name),
                scale_name,
                zp_name,
            )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
            )
            self.quantized_value_map[tensor_name] = quantized_value

    def _quantize_normal_tensors(self):
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map.keys():
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_qdq_pair_for_initializer(initializer, tensor_info.tensor_type, tensor_info.axis)
                else:
                    used_scale, used_zp = self.find_quant_scale_zp(tensor_name)
                    data_found, scale_name, zp_name, _, _ = self._get_quantization_params(
                        tensor_name, used_scale, used_zp
                    )

                    if not data_found:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to "
                            "be quantized are required."
                        )

                    self._add_qdq_pair_for_activation(tensor_name, scale_name, zp_name)

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self):
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
                tensor_provider_name = tensor_info.quant_para_provider
                if tensor_provider_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[tensor_provider_name]
                    # Quantize the input
                    initializer = find_by_name(tensor_name, self.model.initializer())
                    if initializer is not None:
                        raise ValueError("Quantization parameter shared mode is not supported for weight yet")
                    self._add_qdq_pair_for_activation(tensor_name, quantized_value.scale_name, quantized_value.zp_name)

    def _quantize_bias_tensors(self):
        for bias_name, input_name, weight_name, beta in self.bias_to_quantize:
            if bias_name in self.quantized_value_map.keys():
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, input_name, weight_name, beta)
            self.model.remove_initializer(find_by_name(bias_name, self.model.initializer()))
            quant_value = self.quantized_value_map[bias_name]
            inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
            node_name = add_dequant_suffix(bias_name)
            if quant_value.axis is not None:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                    axis=quant_value.axis,
                )
            else:
                dequant_node = onnx.helper.make_node(
                    "DequantizeLinear",
                    inputs,
                    [bias_name],
                    node_name,
                )
            self.model.add_node(dequant_node)

    def is_tensor_quantized(self, tensor_name):
        return tensor_name in self.tensors_to_quantize or tensor_name in self.bias_to_quantize
