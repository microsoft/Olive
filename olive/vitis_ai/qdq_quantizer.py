#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np
import logging
import onnx
import onnx.numpy_helper
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.onnx_quantizer import tensor_proto_to_array
from onnxruntime.quantization.registry import CreateQDQQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer, QDQTensorQuantInfo
from onnxruntime.quantization.quant_utils import (
    DEQUANT_OP_NAME,
    QUANT_OP_NAME,
    QuantizedValue,
    QuantizedValueType,
    __producer__,
    __version__,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_output_suffix,
    add_quant_suffix,
    find_by_name,
)
from olive.vitis_ai.refine import adjust_quantize_info
from olive.vitis_ai.quant_utils import (
    vitis_quantize_data,)


class VitisQuantizer(QDQQuantizer):

    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        input_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        calibrate_method,
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
            input_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.calibrate_method = calibrate_method

        if per_channel:
            logging.error(
                "per_channel is not supported in PowerOfTwoMethod calibrate_method."
            )

        # In PowerOfTwoMethod calibrate_method, QDQ should always appear as a pair.
        # Therefore, we need to add qdq pair to weight.
        if "AddQDQPairToWeight" in self.extra_options and not self.extra_options[
                "AddQDQPairToWeight"]:
            logging.error(
                "AddQDQPairToWeight should be True in PowerOfTwoMethod calibrate_method."
            )
        self.add_qdq_pair_to_weight = True

        # In PowerOfTwoMethod calibrate_method, QDQ should always set WeightSymmetric as True.
        if "WeightSymmetric" in self.extra_options and not self.extra_options[
                "WeightSymmetric"]:
            logging.error(
                "WeightSymmetric should be True in PowerOfTwoMethod calibrate_method."
            )
        self.is_weight_symmetric = True

        # In PowerOfTwoMethod calibrate_method, QDQ should always always set ActivationSymmetric as True.
        if "ActivationSymmetric" in self.extra_options and not self.extra_options[
                "ActivationSymmetric"]:
            logging.error(
                "ActivationSymmetric should be True in PowerOfTwoMethod calibrate_method."
            )
        self.is_activation_symmetric = True

    def vitis_quantize_initializer(self,
                                   weight,
                                   bit_width=8,
                                   keep_float_weight=False):

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
            weight_data.flatten(), bit_width, method=self.calibrate_method)
        scale_initializer = onnx.helper.make_tensor(
            scale_name, onnx_proto.TensorProto.FLOAT, [], [scale])
        zero_initializer = onnx.helper.make_tensor(zp_name,
                                                   onnx_proto.TensorProto.INT8,
                                                   [], [zero_point])
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

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(
                            node)

        self._quantize_normal_tensors()

        self._quantize_sharing_param_tensors()
        self._quantize_refine()

        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__

        return self.model.model

    def _add_qdq_pair_for_initializer(self,
                                      weight_proto,
                                      tensor_type,
                                      axis=None):
        weight_name = weight_proto.name
        q_weight_name, zp_name, scale_name = self.vitis_quantize_initializer(
            weight_proto, self.weight_qType, keep_float_weight=True)

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name,
                                              weight_dequant_output)
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

    def quantize_bias_tensor(self,
                             bias_name,
                             input_name,
                             weight_name,
                             beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                # Use int8 quantization for bias as well as weights.
                self.tensors_to_quantize[bias_name] = QDQTensorQuantInfo()
        else:
            logging.warning("Expected {} to be a weight".format(bias_name))

    def _quantize_refine(self):
        self.model = adjust_quantize_info(self.model,
                                          adjust_vitis_sigmoid=True,
                                          adjust_shift_cut=True,
                                          adjust_shift_bias=True,
                                          adjust_shift_read=True,
                                          adjust_shift_write=True,
                                          align_concat=True,
                                          align_pool=True)
