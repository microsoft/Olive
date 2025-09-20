# Phi-4 Model Optimization

This repository demonstrates the optimization of the following models using **post-training quantization (PTQ)** techniques.

1. [Microsoft Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning)

2. [Microsoft Phi-4-reasoning-plus](https://huggingface.co/microsoft/Phi-4-reasoning-plus)

3. [Microsoft Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning)

4. [Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

## **PTQ Weight Compression for Intel® NPUs/GPUs using Optimum Intel®**

- [**Intel® NPU/GPU**](./openvino/): Optimization with Optimum Intel® on Intel® NPU/GPU to generate an ONNX OpenVINO IR Encapsulated Model instructions are in the [openvino](./openvino/) folder.

## **Optimization and Quantization for AMD NPU**

- [**AMD NPU**](./vitisai/): Instructions to run quantization and optimization for AMD NPU are in the in the [vitisai](./vitisai/) folder.

## **Optimization and Quantization for Qualcomm NPU**

- [**QUALCOMM NPU**](../phi3_5/README.md): Instructions to run quantization and optimization for QUALCOMM NPU are in [this folder](../phi3_5/README.md).
  - Config for phi4 mini instruct model present in qnn/phi4_mini_instruct_qnn_config.json
