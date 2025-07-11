# ResNet Optimization
This folder contains examples of ResNet optimization using different workflows.
- QDQ: [with ONNX Runtime optimizations and static quantization with QDQ format](#resnet-optimization-with-ptq-qdq)
- CPU: [with ONNX Runtime optimizations and static/dynamic quantization](#resnet-optimization-with-ptq-on-cpu)
- CPU: [with PyTorch QAT Default Training Loop and ORT optimizations](#resnet-optimization-with-qat-default-training-loop-on-cpu)
- CPU: [with PyTorch QAT PyTorch Lightning Module and ORT optimizations](#resnet-optimization-with-qat-pytorch-lightning-module-on-cpu)
- AMD DPU: [with AMD Vitis-AI Quantization](#resnet-optimization-with-vitis-ai-ptq-on-amd-dpu)
- Intel GPU: [with OpenVINO and DirectML execution providers in ONNX Runtime](#resnet-optimization-with-openvino-and-dml-execution-providers)
- Qualcomm NPU: [with QNN execution provider in ONNX Runtime](./qnn/)
- Intel® NPU: [Optimization with OpenVINO on Intel® NPU to generate an ONNX OpenVINO IR Encapsulated Model](./openvino/)
- AMD NPU: [Optimization and Quantization with QDQ format for AMD NPU (VitisAI)](#optimization-and-quantization-for-amd-npu)
- Nvidia GPU:[With Nvidia TensorRT-RTX execution provider in ONNX Runtime](#resnet-optimization-with-nvidia-tensorrt-rtx-execution-provider)

Go to [How to run](#how-to-run)

## Optimization Workflows
### ResNet optimization with PTQ QDQ format
This workflow performs ResNet optimization . It performs the pipeline:
- *PyTorch Model -> Onnx Model -> QDQ Quantized Onnx Model -> ONNX Runtime performance tuning*

Config file: [resnet_ptq_qdq.json](resnet_ptq_qdq.json)

#### Accuracy / latency

| Model Version         | Accuracy (Top-1)    | Latency (ms/sample)  | Dataset  |
|-----------------------|---------------------|----------------------|----------|
| PyTorch FP32          | 81.2%               | 2599                 | Imagenet |
| ONNX INT8 (QDQ)       | 78.1%               | 74.7                 | Imagenet |

*Note: Latency can vary significantly depending on the CPU hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*

### Optimization and Quantization for AMD NPU

 This workflow quantizes the model. It performs the pipeline:
 - *HF Model-> ONNX Model -> Optimizations -> Quantized Onnx Model*

 Config file for VitisAI:
 - [microsoft/resnet-50](resnet_ptq_qdq_vitis_ai.json)

### ResNet optimization with PTQ on CPU
This workflow performs ResNet optimization on CPU with ONNX Runtime PTQ. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*
Note that: this case also demonstrates how to leverage the dataset hosted in [AML Datastore](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-datastore?view=azureml-api-2&tabs=cli-identity-based-access%2Ccli-adls-identity-based-access%2Ccli-azfiles-account-key%2Ccli-adlsgen1-identity-based-access). User can set correct local file/folder path or aml datastore url for `data_dir`.

Config file: [resnet_ptq_cpu.json](resnet_ptq_cpu.json)

#### Static Quantization
The workflow in [resnet_static_ptq_cpu.json](resnet_static_ptq_cpu.json) is similar to the above workflow, but specifically uses static quantization instead of static/dynamic quantization.

#### Dynamic Quantization
The workflow in [resnet_dynamic_ptq_cpu.json](resnet_dynamic_ptq_cpu.json) is similar to the above workflow, but specifically uses dynamic quantization instead of static/dynamic quantization.

### ResNet optimization with QAT Default Training Loop on CPU
This workflow performs ResNet optimization on CPU with QAT Default Training Loop. It performs the optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> ONNX Runtime performance tuning*

Config file: [resnet_qat_default_train_loop_cpu.json](resnet_qat_default_train_loop_cpu.json)

### ResNet optimization with QAT PyTorch Lightning Module on CPU
This workflow performs ResNet optimization on CPU with QAT PyTorch Lightning Module. It performs the optimization pipeline:
- *PyTorch Model -> PyTorch Model after QAT -> Onnx Model -> ONNX Runtime performance tuning*

Config file: [resnet_qat_lightning_module_cpu.json](resnet_qat_lightning_module_cpu.json)

### ResNet optimization with Vitis-AI PTQ on AMD DPU
This workflow performs ResNet optimization on AMD DPU with AMD Vitis-AI Quantization. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> AMD Vitis-AI Quantized Onnx Model*

Config file: [resnet_vitis_ai_ptq_cpu.json](resnet_vitis_ai_ptq_cpu.json)

### ResNet optimization with Nvidia TensorRT-RTX execution provider
This example performs ResNet optimization with Nvidia TensorRT-RTX execution provider. It performs the optimization pipeline:
- *ONNX Model -> fp16 Onnx Model*

Config file: [resnet_trtrtx.json](resnet_trtrtx.json)

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
olive run --config <config_file>.json --setup
```

Then, optimize the model
```
olive run --config <config_file>.json
```

After running the above command, the final model will be saved in the *output_dir* specified in the config file.
