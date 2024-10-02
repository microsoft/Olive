---
hide:
- navigation
- toc
---

# Overview

## What is OLIVE?

<div class="result" markdown>

![Image title](images/olive-flow.png){ width="600" align=right}

OLIVE (**O**NNX **LIVE**) is a cutting edge model optimization tookit with accompanying CLI that enables you to ship models for the [ONNX runtime](https://onnxruntime.ai) with quality and performance. 

The input to OLIVE is typically a PyTorch or Hugging Face model and the output is an optimized ONNX model that is executed on a device running the ONNX runtime that has an AI accelerator (NPU, GPU, CPU) provided by a hardware vendor such as Qualcomm, AMD, Nvidia or Intel. 

OLIVE executes a *workflow*, which is a set of individual model optimization tasks (for example, compression, graph capture, quantization, graph optimization) - called *passes* - that execute in a specific order. Each pass has a set of parameters that can be tuned to achieve the best metrics, say accuracy and latency, that are evaluated by the respective *evaluator*.
OLIVE employs a *search strategy* that uses a *search algorithm* to auto-tune each pass one by one or set of passes together.

</div>


## Benefits of using OLIVE

- [x] No more trial-and-error of manually experimenting with different graph optimizations, compression and quantization. Define your quality and performance constraints and let OLIVE automatically find the best model.
- [x] 40+ built-in model optimization components covering cutting edge techniques in quantization, compression, graph optimization and finetuning.
- [x] Easy-to-use CLI for common model optimization tasks. For example, `olive quantize`, `olive auto-opt`, `olive finetune`.
- [x] Model packaging and deployment built-in.
- [x] Supports *Multi LoRA* serving.
- [x] Construct *workflows* using YAML/JSON to *orchestrate* model optimization and deployment tasks.
- [x] Hugging Face and Azure AI Integration.
- [x] Built-in caching mechanism to save costs.

## Try OLIVE

[:octicons-arrow-right-24: Getting started](getting-started/getting-started.md)