# AutoAwq in Onnxruntime
Sample use case of Olive to optimize facebook/opt_125m model using AutoAwq which produces a quantized model that can be converted and run in Onnxruntime.

## Optimization Workflows
Performs optimization pipeline:
- *Pytorch Model -> AutoAwqQuantization -> Onnx Model -> Transformers Optimized*

## Prerequisite

Before running this script, you need to install the required python packages from the requirements.txt file.
```bash
pip install -r requirements.txt
```

Also, please ensure you already installed olive-ai. Please refer to the [installation guide](https://github.com/microsoft/Olive?tab=readme-ov-file#installation) for more information.

## Optimization Usage

Olive already simplifies the optimization process by providing a single json config to run the optimization workflow. You can run the following command to execute the workflow:

```bash
python -m olive.workflows.run --config awq.json
```
