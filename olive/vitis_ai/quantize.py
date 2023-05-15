#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path

import onnx
import onnx.helper as helper
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quantize import quantize_static as ort_quantize_static
from onnxruntime.quantization.quant_utils import QuantizationMode, QuantType, load_model, QuantFormat
from onnxruntime.quantization.registry import QLinearOpsRegistry

from olive.vitis_ai.calibrate import create_calibrator_power_of_two, PowerOfTwoMethod
from olive.vitis_ai.qdq_quantizer import VitisQuantizer
from olive.vitis_ai.quant_utils import get_exclude_nodes


def quantize_static(
    model_input,
    model_output,
    calibration_data_reader: CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    input_nodes=[],
    output_nodes=[],
    op_types_to_quantize=[],
    per_channel=False,
    reduce_range=False,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    optimize_model=True,
    use_external_data_format=False,
    calibrate_method=PowerOfTwoMethod.MinMSE,
    extra_options={},
):
    """
    Given an onnx model and calibration data reader, create a quantized onnx model and save it into a file

    Args:

        model_input: file path of model to quantize
        model_output: file path of quantized model
        calibration_data_reader: a calibration data reader. It
            enumerates calibration data and generates inputs for the
            original model.
        quant_format: QuantFormat{QOperator, QDQ}.
            QOperator format quantizes the model with quantized operators directly.
            QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
        activation_type:
            quantization data type of activation. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        calibrate_method:
            Current calibration methods supported are MinMax and Entropy and PowerOfTwoMethod of NonOverflow and MinMSE.
                Please use CalibrationMethod.MinMax or CalibrationMethod.Entropy 
                or PowerOfTwoMethod.MinMSE or PowerOfTwoMethod.NonOverflow as options.
                Default is PowerOfTwoMethod.MinMSE.
        input_nodes:
            The list of nodes at which quantization begins in a model.
                The nodes prior to these will not be quantized.
                By default, this is empty.
        output_nodes:
            The list of nodes at which quantization ends in a model.
                The nodes following to these will not be quantized.
                By default, this is empty.
        op_types_to_quantize:
                specify the types of operators to quantize, like ['Conv'] to quantize Conv only.
                It quantizes all supported operators by default.
        per_channel: quantize weights per channel, PowerOfTwoMethod do not support per_channel now.
        reduce_range:
            quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine,
            especially for per-channel mode
        weight_type:
            quantization data type of weight. Please refer to
            https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
        nodes_to_quantize:
            List of nodes names to quantize. When this list is not None only the nodes in this list
            are quantized.
            example:
            [
                'Conv__224',
                'Conv__252'
            ]
        nodes_to_exclude:
            List of nodes names to exclude. The nodes in this list will be excluded from quantization
            when it is not None.
        optimize_model: Deprecating Soon! Optimize model before quantization. NOT recommended, optimization will
            change the computation graph, making debugging of quantization loss difficult.
        use_external_data_format: option used for large size (>2GB) model. Set to False by default.
        extra_options:
            key value pair dictionary for various options in different case. Current used:
                extra.Sigmoid.nnapi = True/False  (Default is False)
                ActivationSymmetric = True/False: symmetrize calibration data for activations
                                                  (In PowerOfTwoMethod calibrate_method, default is True.
                                                  In MinMax and Entropy calibrate_method, default is False).
                WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
                EnableSubgraph = True/False : Default is False. If enabled, subgraph will be quantized.
                                              Dyanmic mode currently is supported. Will support more in the future.
                ForceQuantizeNoInputCheck = True/False :
                    By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                    quantized already. Setting to True to force such operator always quantize input and so generate
                    quantized output. Also, the True behavior could be disabled per node using the nodes_to_exclude.
                MatMulConstBOnly = True/False:
                    Default is False for static mode. If enabled, only MatMul with const B will be quantized.
                AddQDQPairToWeight = True/False :
                    In PowerOfTwoMethod calibrate_method, default is True. It remains floating-point weight and
                    inserts both QuantizeLinear/DeQuantizeLinear nodes to weight.
                    In MinMax and Entropy calibrate_method, default is False which quantizes floating-point weight
                    and feeds it to solely inserted DeQuantizeLinear node.
                OpTypesToExcludeOutputQuantization = list of op type :
                    Default is []. If any op type is specified, it won't quantize the output of ops with this
                    specific op types.
                DedicatedQDQPair = True/False :
                    Default is False. When inserting QDQ pair, multiple nodes can share a single QDQ pair as their
                    inputs. If True, it will create identical and dedicated QDQ pair for each node.
                QDQOpTypePerChannelSupportToAxis = dictionary :
                    Default is {}. Set channel axis for specific op type, for example: {'MatMul': 1}, and it's
                    effective only when per channel quantization is supported and per_channel is True. If specific
                    op type supports per channel quantization but not explicitly specified with channel axis,
                    default channel axis will be used.
                CalibTensorRangeSymmetric = True/False :
                    Default is False. If enabled, the final range of tensor during calibration will be explicitly
                    set to symmetric to central point "0".
                CalibMovingAverage = True/False :
                    Default is False. If enabled, the moving average of the minimum and maximum values will be
                    computed when the calibration method selected is MinMax.
                CalibMovingAverageConstant = float :
                    Default is 0.01. Constant smoothing factor to use when computing the moving average of the
                    minimum and maximum values. Effective only when the calibration method selected is MinMax and
                    when CalibMovingAverage is set to True.
    """

    if calibrate_method in CalibrationMethod:
        return ort_quantize_static(
            model_input, model_output, calibration_data_reader, quant_format,
            op_types_to_quantize, per_channel, reduce_range, activation_type,
            weight_type, nodes_to_quantize, nodes_to_exclude, optimize_model,
            use_external_data_format, calibrate_method, extra_options)

    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(QLinearOpsRegistry.keys())

    model = load_model(Path(model_input), optimize_model)

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        calibrator = create_calibrator_power_of_two(
            model,
            op_types_to_quantize,
            augmented_model_path=Path(quant_tmp_dir).joinpath(
                "augmented_model.onnx").as_posix(),
            method=calibrate_method,
            use_external_data_format=use_external_data_format,
            extra_options=None,
        )

        calibrator.collect_data(calibration_data_reader)
        tensors_range = calibrator.compute_range()
    if input_nodes or output_nodes:
        if nodes_to_exclude:
            nodes_to_exclude += get_exclude_nodes(model_input, input_nodes, output_nodes)
        else:
            nodes_to_exclude = get_exclude_nodes(model_input, input_nodes, output_nodes)

    if quant_format is QuantFormat.QOperator:
        quantizer = ONNXQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
    elif quant_format is QuantFormat.QDQ:
        quantizer = VitisQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            extra_options,
        )
    else:
        raise TypeError(
            "Invalid quant_format type, it must be either QuantFormat or VitisQuantFormat."
        )

    quantizer.quantize_model()

    quantizer.model.save_model_to_file(model_output, use_external_data_format)
