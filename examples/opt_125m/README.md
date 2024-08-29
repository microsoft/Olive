# AutoAwq in Onnxruntime
Sample use case of Olive to optimize facebook/opt_125m model using AutoAwq/AutoGPTQ which produces a quantized model that can be converted and run in Onnxruntime.

## Optimization Workflows
Performs optimization pipeline:
- *Pytorch Model -> AutoAwqQuantization -> Pytorch Model*
- *Pytorch Model -> AutoAwqQuantization -> Onnx Model -> Transformers Optimized*
- *Pytorch Model -> AutoGPTQQuantization -> Pytorch Model*
- *Pytorch Model -> AutoGPTQQuantization -> Onnx Model -> Transformers Optimized*

## Prerequisite

Before running this script, you need to install the required python packages from the requirements.txt file.
```bash
# awq
pip install -r requirements-awq.txt

# gptq
pip install -r requirements-gptq.txt
```

Also, please ensure you already installed olive-ai. Please refer to the [installation guide](https://github.com/microsoft/Olive?tab=readme-ov-file#installation) for more information.

## Optimization Usage

Olive already simplifies the optimization process by providing a single json config to run the optimization workflow. You can run the following command to execute the workflow:

```bash
# only run awq
python -m olive.workflows.run --config awq.json

# run awq, export to onnx, and optimize onnx
python -m olive.workflows.run --config awq_onnx.json

# only run gptq
python -m olive.workflows.run --config gptq.json

# run gptq, export to onnx, and optimize onnx
python -m olive.workflows.run --config gptq_onnx.json
```
