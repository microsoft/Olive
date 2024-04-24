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

from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.quantize import quantize_static as ort_quantize_static
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

from olive.passes.onnx.vitis_ai.calibrate import PowerOfTwoMethod, create_calibrator_power_of_two
from olive.passes.onnx.vitis_ai.quant_utils import get_exclude_nodes, is_ort_version_below_1_16
from olive.passes.onnx.vitis_ai.quantizer import VitisDPUQuantizer, VitisQDQQuantizer, VitisQOpQuantizer

# pylint: skip-file
# ruff: noqa


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
    optimize_model=False,
    use_external_data_format=False,
    calibrate_method=PowerOfTwoMethod.MinMSE,
    need_layer_fusing=False,
    execution_providers=["CPUExecutionProvider"],
    enable_dpu=False,
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
        need_layer_fusing: This parameter determines whether to perform layer fusion on certain operations
            (such as conv-relu) in the network, The default value is False.
        execution_providers: This parameter specifies the execution providers to run the network, The default
            value is ['CPUExecutionProvider'].
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
        ort_quantize_args = {
            "model_input": model_input,
            "model_output": model_output,
            "calibration_data_reader": calibration_data_reader,
            "quant_format": quant_format,
            "op_types_to_quantize": op_types_to_quantize,
            "per_channel": per_channel,
            "reduce_range": reduce_range,
            "activation_type": activation_type,
            "weight_type": weight_type,
            "nodes_to_quantize": nodes_to_quantize,
            "nodes_to_exclude": nodes_to_exclude,
            "use_external_data_format": use_external_data_format,
            "calibrate_method": calibrate_method,
            "extra_options": extra_options,
        }
        # for ORT version < 1.16.0, set optimize_model to False
        # always set it to False since it is not recommended and is removed in ORT 1.16.0
        # user needs to call pre-process to optimize the model
        if is_ort_version_below_1_16():
            ort_quantize_args["optimize_model"] = False
        ort_quantize_static(**ort_quantize_args)
        return

    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        q_linear_ops = list(QLinearOpsRegistry.keys())
        qdq_ops = list(QDQRegistry.keys())
        op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    # for ORT version >= 1.16.0, we should use load_model_with_shape_infer
    # since load_model is already removed in ORT 1.16.
    if is_ort_version_below_1_16():
        from onnxruntime.quantization.quant_utils import load_model

        model = load_model(Path(model_input), optimize_model)
    else:
        from onnxruntime.quantization.quant_utils import load_model_with_shape_infer

        model = load_model_with_shape_infer(Path(model_input))

    calib_extra_options_keys = [
        ("ActivationSymmetric", "symmetric"),
    ]

    calib_extra_options = {
        key: extra_options.get(name) for (name, key) in calib_extra_options_keys if name in extra_options
    }

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        calibrator = create_calibrator_power_of_two(
            Path(model_input),
            op_types_to_quantize,
            augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
            activation_type=QuantType.QInt8,
            method=calibrate_method,
            use_external_data_format=use_external_data_format,
            execution_providers=execution_providers,
            extra_options=calib_extra_options,
        )

        calibrator.collect_data(calibration_data_reader)
        if is_ort_version_below_1_16():
            tensors_range = calibrator.compute_range()
        elif calibrate_method == PowerOfTwoMethod.MinMSE:
            tensors_range = calibrator.compute_range()
            from onnxruntime.quantization.calibrate import TensorsData

            tensors_range = TensorsData(CalibrationMethod.MinMax, tensors_range)
        else:
            tensors_range = calibrator.compute_data()
        del calibrator

    if input_nodes or output_nodes:
        if nodes_to_exclude:
            nodes_to_exclude += get_exclude_nodes(model_input, input_nodes, output_nodes)
        else:
            nodes_to_exclude = get_exclude_nodes(model_input, input_nodes, output_nodes)

    if quant_format is QuantFormat.QOperator:
        quantizer = VitisQOpQuantizer(
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
    elif quant_format is QuantFormat.QDQ and not enable_dpu:
        quantizer = VitisQDQQuantizer(
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
    elif quant_format is QuantFormat.QDQ and enable_dpu:
        quantizer = VitisDPUQuantizer(
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
            need_layer_fusing,
            extra_options,
        )
    else:
        raise TypeError("Invalid quant_format type, it must be QuantFormat.")

    quantizer.quantize_model()

    quantizer.model.save_model_to_file(model_output, use_external_data_format)
