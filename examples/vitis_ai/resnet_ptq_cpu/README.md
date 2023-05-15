# ResNet quantization with Vitis AI ONNX PTQ on CPU
This folder contains a sample use case of Olive to quantize a **ONNX Float Model** using Vitis AI ONNX quantization.

For optimize a ResNet model using onnx conversion and onnx dynamic/static quantization tuner in Olive, you can refer to [ResNet optimization with PTQ on CPU](../../resnet_ptq_cpu/README.md). In this document, we will introduce how to use the Vitis AI ONNX Quantization Pass in Olive to quantize an ONNX float model.

Vitis AI ONNX Quantization pipeline:
- *ONNX Float Model -> Vitis AI ONNX quantization -> Quantized ONNX Model (for ONNXRUNTIME/VITIS AI EP)*

Outputs a quantized ONNX model.

## Prerequisites
### Installation and preparing floating-point ONNX models and calibration data.
```
git clone https://github.com/microsoft/Olive
cd Olive
pip install -e .
```

Please note that vitis_ai requires a dependency on onnxruntime >=1.14.

## Configuring

You can refer to "[resnet_config.json](./resnet_config.json)" to configure the ONNX float model you need to quantize. Note that if you use the Vitis AI quantization tool, you need to specify **"passes"** as **"vitis_ai_quantization"**, and specify **"type"** as **"VitisQuantization"**.

```
{
    "verbose": true,
    "input_model":{
        "type": "ONNXModel",
        "config": {
            "model_path": "models/resnet18.onnx",
            "is_file": true
        }
    },
    "systems": {
        "local_system": {"type": "LocalSystem"}
    },
    "evaluators": {
        "common_evaluator":{
            "metrics":[
            ],
            "target": "local_system"
        }
    },
    "passes": {
        "vitis_ai_quantization": {
            "type": "VitisQuantization",
            "disable_search": true,
            "config": {
                "user_script": "user_script.py",
                "data_dir": "data/resnet",
                "model_path": "models/resnet18.onnx",
                "dataloader_func": "resnet_calibration_reader"
            },
            "clean_run_cache": false
        }
    },
    "engine": {
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "exhaustive"
        },
        "evaluator": "common_evaluator",
        "host": {"type": "LocalSystem"},
        "cache_dir": "cache"
    }
}
```



## Running sample Vitis AI ONNX quantization using config

```
python -m olive.workflows.run --config resnet_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("resnet_config.json")
```

The quantized model will be saved at the following location:

cache/models/0_VitisQuantization-*/model.onnx

## License

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
