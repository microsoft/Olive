# QNN

Qualcomm AI Engine Direct is a Qualcomm Technologies Inc. software architecture for AI/ML use cases on Qualcomm chipsets and AI acceleration cores.

Olive provides tools to convert models from different frameworks such as ONNX, TensorFlow, and PyTorch to QNN model formats and quantize them to 8 bit fixed point for running on NPU cores.
Olive uses the development tools available in the [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk) (QNN SDK).

## Prerequisites
### Download and unzip QNN SDK
Download the QNN SDK and unzip the file.

Set the environment variable QNN_SDK_ROOT as <qnn-sdk-unzipped-path>.

### Configure Olive QNN
```bash
olive configure-qualcomm-sdk --py_version 3.8 --sdk qnn
```
**Note:** If `olive` cannot be found in your path, you can use `python -m olive` instead.

## Model Conversion/Quantization
`QNNConversion` converts ONNX, TensorFlow, or PyTorch models to QNN C++ model. Optionally, it can also quantize the model if a calibration dataset is provided using the `--input_list` in the extra_args parameter.

The C++ model must be compiled into a model library for the desired target using the `QNNModelLibGenerator` pass for inference on the target device.

Please refer to [QNNConversion](qnn_conversion) for more details about the pass and its config parameters.

### Example Configuration
**Conversion**
```json
{
    "type": "QNNConversion"
}
```

**Conversion and Quantization**
```json
{
    "type": "QNNConversion",
    "extra_args": "--input_list <input_list.txt>"
}
```

## Model Library Generation
`QNNModelLibGenerator` compiles the QNN C++ model into a model library for the desired target. The model library can be used for inference on the target device.

Please refer to [QNNModelLibGenerator](qnn_model_lib_generator) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "QNNModelLibGenerator",
    "lib_targets": "x86_64-linux-clang"
}
```

## Context Binary Generation
A QNN Context provides execution environment for graphs and operations. Context content can be cached into a binary form which later can be used for faster context/graph loading.
`QNNContextBinaryGenerator` generated the context binary from a compiled model library using a specific backend.

Please refer to [QNNContextBinaryGenerator](qnn_context_binary_generator) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "QNNContextBinaryGenerator",
    "backend": "<QNN_BACKEND.so>"
}
```
