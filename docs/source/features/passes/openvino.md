# OpenVINO

OpenVINO is a cross-platform deep learning toolkit developed by Intel. The name stands for "Open Visual Inference and Neural Network
Optimization." OpenVINO focuses on optimizing neural network inference with a write-once, deploy-anywhere approach for Intel hardware
platforms.

Read more at: [Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)


## Prerequisites
Note: OpenVINO version in Olive: 2022.3.0

### Option 1: install Olive with OpenVINO extras
```
pip install olive-ai[openvino]
```

### Option 2: Install OpenVINO Runtime and OpenVINO Development Tools from Pypi
```
pip install openvino==2022.3.0 openvino-dev[tensorflow,onnx]==2022.3.0
```


## Model Conversion
`OpenVINOConversion` pass will convert the model from original framework to OpenVino IR Model. `PyTorchModelHandler`, `ONNXModelHandler` and
`TensorFlowModelHandler` are supported for now.

Please refer to [OpenVINOConversion](openvino_conversion) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "OpenVINOConversion",
    "config": {
        "input_shape": [1, 3, 32, 32]
    }
}
```

## Post Training Quantization (PTQ)
`OpenVINOQuantization` pass will run Post-training quantization for OpenVINO model which supports the uniform integer quantization method.
This method allows moving from floating-point precision to integer precision (for example, 8-bit) for weights and activations during the
inference time. It helps to reduce the model size, memory footprint and latency, as well as improve the computational efficiency, using
integer arithmetic. During the quantization process the model undergoes the transformation process when additional operations, that contain
quantization information, are inserted into the model. The actual transition to integer arithmetic happens at model inference.

Please refer to [OpenVINOQuantization](openvino_quantization) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "OpenVINOQuantization",
    "config": {
        "engine_config": {"device": "CPU", "stat_requests_number": 2, "eval_requests_number": 2},
        "algorithms": [
            {
                "name": "DefaultQuantization",
                "params": {"target_device": "CPU", "preset": "performance", "stat_subset_size": 300},
            }
        ],
        "data_dir": "data_dir",
        "user_script": "user_script.py",
        "dataloader_func": "create_dataloader",
    }
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/cifar10_openvino_intel_hw/user_script.py)
for an example implementation of `"user_script.py"` and `"create_dataloader"`.
