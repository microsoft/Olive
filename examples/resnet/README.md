# ResNet Optimization
This folder contains examples of ResNet optimization using different workflows.

## Optimization Workflows
### ResNet optimization with PTQ on CPU
This workflow performs ResNet optimization on CPU with ONNX Runtime PTQ. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

Config file: [resnet_ptq_cpu.json](resnet_ptq_cpu.json)

#### Static Quantization
The workflow in [resnet_static_ptq_cpu.json](resnet_static_ptq_cpu.json) is similar to the above workflow, but specifically uses static quantization instead of static/dynamic quantization.

#### Dynamic Quantization
The workflow in [resnet_dynamic_ptq_cpu.json](resnet_dynamic_ptq_cpu.json) is similar to the above workflow, but specifically uses dynamic quantization instead of static/dynamic quantization.


### ResNet optimization with Vitis-AI PTQ on CPU
This workflow performs ResNet optimization on CPU with AMD Vitis-AI Quantization. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> AMD Vitis-AI Quantized Onnx Model*

Config file: [resnet_vitis_ai_ptq_cpu.json](resnet_vitis_ai_ptq_cpu.json)

### ResNet optimization with QAT Default Training Loop on CPU
This workflow performs ResNet optimization on CPU with QAT Default Training Loop. It performs the optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> ONNX Runtime performance tuning*

Config file: [resnet_qat_default_train_loop_cpu.json](resnet_qat_default_train_loop_cpu.json)

### ResNet optimization with QAT PyTorch Lightning Module on CPU
This workflow performs ResNet optimization on CPU with QAT PyTorch Lightning Module. It performs the optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> ONNX Runtime performance tuning*

Config file: [resnet_qat_lightning_module_cpu.json](resnet_qat_lightning_module_cpu.json)

## How to run
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Prepare data and model
```
python prepare_model_data.py
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.
```
python -m olive.workflows.run --config <config_file>.json --setup
```

Then, optimize the model
```
python -m olive.workflows.run --config <config_file>.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("<config_file>.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
