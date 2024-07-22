# OpenVINO

OpenVINO is a cross-platform deep learning toolkit developed by Intel. The name stands for "Open Visual Inference and Neural Network
Optimization." OpenVINO focuses on optimizing neural network inference with a write-once, deploy-anywhere approach for Intel hardware
platforms.

Read more at: [Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)


## Prerequisites
Note: OpenVINO version in Olive: 2023.2.0

### Option 1: install Olive with OpenVINO extras
```bash
pip install olive-ai[openvino]
```

### Option 2: Install OpenVINO Runtime and OpenVINO Development Tools from Pypi
```bash
pip install openvino==2023.2.0
```


## Model Conversion
`OpenVINOConversion` pass will convert the model from original framework to OpenVino IR Model. `PyTorchModelHandler`, `ONNXModelHandler` and
`TensorFlowModelHandler` are supported for now.

Please refer to [OpenVINOConversion](openvino_conversion) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "OpenVINOConversion",
    "input": [1, 3, 32, 32]
}
```

## Post Training Quantization (PTQ)
`OpenVINOQuantization` pass will run [Post-training quantization](https://docs.openvino.ai/2023.3/ptq_introduction.html) for OpenVINO model which supports the uniform integer quantization method.
This method allows moving from floating-point precision to integer precision (for example, 8-bit) for weights and activations during the
inference time. It helps to reduce the model size, memory footprint and latency, as well as improve the computational efficiency, using
integer arithmetic. During the quantization process the model undergoes the transformation process when additional operations, that contain
quantization information, are inserted into the model. The actual transition to integer arithmetic happens at model inference.

Please refer to [OpenVINOQuantization](openvino_quantization) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "OpenVINOQuantizationWithAccuracy",
    "data_dir": "data",
    "user_script": "user_script.py",
    "dataloader_func": "create_dataloader",
    "validation_func": "validate",
    "max_drop": 0.01,
    "drop_type": "ABSOLUTE"
}
```
