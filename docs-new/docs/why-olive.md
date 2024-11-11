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

The input to OLIVE is typically a PyTorch or Hugging Face model and the output is an optimized ONNX model that is executed on a device (deployment target) running the ONNX runtime. OLIVE will optimize the model for the deployment target's AI accelerator (NPU, GPU, CPU) provided by a hardware vendor such as Qualcomm, AMD, Nvidia or Intel. 

OLIVE executes a *workflow*, which is an ordered sequence of individual model optimization tasks called *passes* - example passes include: model compression, graph capture, quantization, graph optimization. Each pass has a set of parameters that can be tuned to achieve the best metrics, say accuracy and latency, that are evaluated by the respective *evaluator*. OLIVE employs a *search strategy* that uses a *search algorithm* to auto-tune each pass one by one or set of passes together.

</div>


## Benefits of using OLIVE

- [x] **Reduce frustration and time** of trial-and-error manual experimentation with different techniquies for graph optimization, compression and quantization. Define your quality and performance constraints and let OLIVE automatically find the best model for you.
- [x] **40+ built-in model optimization components** covering cutting edge techniques in quantization, compression, graph optimization and finetuning.
- [x] **Easy-to-use CLI** for common model optimization tasks. For example, `olive quantize`, `olive auto-opt`, `olive finetune`.
- [x] Model packaging and deployment built-in.
- [x] Supports **Multi LoRA serving**.
- [x] Construct *workflows* using YAML/JSON to *orchestrate* model optimization and deployment tasks.
- [x] Hugging Face and Azure AI Integration.
- [x] Built-in **caching** mechanism to save costs.

## Try OLIVE

[:octicons-arrow-right-24: Getting started](getting-started/getting-started.md)