# Phi3.5 Model Optimization for Qualcomm NPU

This repository demonstrates the optimization of the [Microsoft Phi-3.5 Mini Instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model for deployment on Qualcomm NPUs (such as Snapdragon X Series) using the ONNX Runtime QNNExecutionProvider.

To enable efficient deployment on NPU, we perform a series of optimizations, including quantization, ahead-of-time (AOT) compilation, and handling of dynamic and static input shapes. The result is a model compiled for optimal execution on Qualcomm NPUs.

## Table of Contents

1. [Optimization Process](#optimization-process)
2. [Handling Dynamic and Static Input Shapes](#handling-dynamic-and-static-input-shapes)
3. [Resource Optimization Strategy](#resource-optimization-strategy)
4. [Compilation for NPU Deployment](#compilation-for-npu-deployment)
5. [Summary of Key Steps](#summary-of-key-steps)
6. [Requirements](#requirements)
7. [Usage](#usage)

## Optimization Process

The model optimization process includes the following steps:

1. **Weight Rotation with [QuaRot](https://arxiv.org/abs/2404.00456)**: This step adjusts the model’s weights to make it more suitable for quantization.
2. **4-bit Per-Channel Symmetric Quantization with [GPTQ](https://arxiv.org/abs/2210.17323)**: Applies weight-only quantization (4-bit) to the transformer linear layers to reduce model size.
3. **ONNX Graph Capture**: Captures the ONNX graph for subsequent optimizations.
4. **4-bit Block-wise Quantization**: Quantizes the embedding layer and the language modeling head using weight-only 4-bit quantization.
5. **16-bit Activation Quantization**: Quantize activations to 16-bits, balancing efficiency and precision.

The resulting model is a quantized QDQ (Quantize-Dequantize) model with 4-bit weights and 16-bit activations.

## Handling Dynamic and Static Input Shapes

NPUs require pre-compiled graphs, which means the model must use static input shapes. The text generation process consists of two main stages with different input shape requirements:

- **Prefill (Context/Prompt Processing)**: This stage processes multiple tokens simultaneously.
- **Token Generation (Iteration)**: This stage processes one token at a time.

To accommodate these differing requirements, we optimize the model by creating two instances: one for prefill and one for token generation. This allows the model to handle both static and dynamic input shapes effectively.

## Resource Optimization Strategy

To optimize resource usage:

- **Embedding Layer and Language Model Head**: These components are executed on the CPU and can handle dynamic input shapes.
- **Transformer Layers**: These layers are executed on the NPU, which requires static input shapes.
- **Weight Sharing**: The prefill and token generation models share weights (with different input shapes), minimizing memory usage.

Additionally, we use **[GroupQueryAttention](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention)** to enable dynamic key-value caching, which supports long-context processing and token generation.

## Compilation for NPU Deployment

After optimization, the model is compiled for deployment on the Qualcomm NPU using the ONNX Runtime QNNExecutionProvider. The process includes the following steps:

1. **Split the Quantized Model**: Divide the model into three parts: embedding layer, transformer layers, and language model head.
2. **Fix Static Input Shape for Transformer Layers**: Set static input shapes for transformer layers—(1, 64) for prefill (batch size, sequence length) and (1, 1) for token generation.
3. **Compile with ONNX Runtime QNNExecutionProvider**: Use the QNNExecutionProvider to compile the model with weight sharing.

## Summary of Key Steps

- **Model Quantization**: Apply 4-bit weight and 16-bit activation quantization.
- **Prefill and Token Generation Optimization**: Create two instances of the model to handle dynamic and static input shapes.
- **Deployment on Qualcomm NPU**: Use ONNX Runtime QNNExecutionProvider for AOT compilation.

This approach ensures efficient execution on Qualcomm NPUs, optimizing memory and computational resources for text generation tasks.

## Requirements

The optimization process requires several computationally intensive quantization steps, and therefore demands GPU resources. In an [x64 Python environment with olive-ai installed](https://github.com/microsoft/Olive/tree/main/examples#important), the following packages are required:

```bash
# Common requirements
pip install -r requirements.txt

# ONNX Runtime packages
pip install "onnxruntime-gpu>=1.21.0" "onnxruntime-genai-cuda>=0.6.0"

# AutoGPTQ: Install from source (stable package may be slow for weight packing)
BUILD_CUDA_EXT=0 pip install -vvv --no-build-isolation git+https://github.com/PanQiWei/AutoGPTQ.git
```

For AOT compilation, the latest nightly x64 build of onnxruntime-qnn is required. In a separate Python environment with olive-ai installed, the following packages are required:

```bash
# Common requirements
pip install -r requirements.txt

# ONNX Runtime packages
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
```

## Usage

The entire workflow is configured via the [config.json](config.json) file. Update the `/path/to/qnn/env/bin` in the config file to point to the Python environment where `onnxruntime-qnn` is installed.

To begin the optimization process, run the following command in the first Python environment:

```bash
olive run --config config.json
```

The optimization process will take some time to complete. Once finished, the optimized model will be saved in the `models` directory.

*Note*: If the optimization process fails silently during the context binary generation step, simply rerun the command. The process will resume from the last completed step.
