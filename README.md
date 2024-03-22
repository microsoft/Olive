# Olive

Olive is an easy-to-use hardware-aware model optimization tool that composes industry-leading techniques
across model compression, optimization, and compilation. Given a model and targeted hardware, Olive composes the best
suitable optimization techniques to output the most efficient model(s) for inferring on cloud or edge, while taking
a set of constraints such as accuracy and latency into consideration.

Since every ML accelerator vendor implements their own acceleration tool chains to make the most of their hardware, hardware-aware
optimizations are fragmented. With Olive, we can:

Reduce engineering effort for optimizing models for cloud and edge: Developers are required to learn and utilize
multiple hardware vendor-specific toolchains in order to prepare and optimize their trained model for deployment.
Olive aims to simplify the experience by aggregating and automating optimization techniques for the desired hardware
targets.

Build up a unified optimization framework: Given that no single optimization technique serves all scenarios well,
Olive enables an extensible framework that allows industry to easily plugin their optimization innovations.  Olive can
efficiently compose and tune integrated techniques for offering a ready-to-use E2E optimization solution.

## News
- [ Mar 2024 ] [Fine-tune SLM with Microsoft Olive](https://techcommunity.microsoft.com/t5/educator-developer-blog/journey-series-for-generative-ai-application-architecture-fine/ba-p/4080813)
- [ Jan 2024 ] [Accelerating SD Turbo and SDXL Turbo Inference with ONNX Runtime and Olive](https://huggingface.co/blog/sdxl_ort_inference)
- [ Dec 2023 ] [Windows AI Studio - VS Code Extension that uses Olive to fine tune models](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio&ssr=false#overview)
- [ Nov 2023 ] [Elevating the developer experience on Windows with new AI tools and productivity tools](https://blogs.windows.com/windowsdeveloper/2023/11/15/elevating-the-developer-experience-on-windows-with-new-ai-tools-and-productivity-tools/)
- [ Nov 2023 ] [Accelerating LLaMA-2 Inference with ONNX Runtime using Olive](https://onnxruntime.ai/blogs/accelerating-llama-2)
- [ Nov 2023 ] [Olive 0.4.0 released with support for LoRA fine-tuning and Llama2 optimizations](https://github.com/microsoft/Olive/releases/tag/v0.4.0)
- [ Nov 2023 ] [Intel and Microsoft Collaborate to Optimize DirectML for Intel® Arc™ Graphics Solutions using Olive](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Intel-and-Microsoft-Collaborate-to-Optimize-DirectML-for-Intel/post/1542055)
- [ Nov 2023 ] [Running Olive Optimized Llama2 with Microsoft DirectML on AMD Radeon Graphics](https://community.amd.com/t5/ai/how-to-running-optimized-llama2-with-microsoft-directml-on-amd/ba-p/645190)
- [ Oct 2023 ] [AMD Microsoft Olive Optimizations for Stable Diffusion Performance Analysis](https://www.pugetsystems.com/labs/articles/amd-microsoft-olive-optimizations-for-stable-diffusion-performance-analysis/)
- [ Sep 2023 ] [Running Optimized Automatic1111 Stable Diffusion WebUI on AMD GPUs](https://community.amd.com/t5/ai/updated-how-to-running-optimized-automatic1111-stable-diffusion/ba-p/630252)
- [ Jul 2023 ] [Build accelerated AI apps for NPUs with Olive](https://www.infoworld.com/article/3701452/build-accelerated-ai-apps-for-npus-with-olive.html)
- [ Jun 2023 ] [Olive: A user-friendly toolchain for hardware-aware model optimization](https://cloudblogs.microsoft.com/opensource/2023/06/26/olive-a-user-friendly-toolchain-for-hardware-aware-model-optimization/)
- [ May 2023 ] [Optimize DirectML performance with Olive](https://devblogs.microsoft.com/directx/optimize-directml-performance-with-olive/)
- [ May 2023 ] [Optimize Stable Diffusion Using Olive](https://devblogs.microsoft.com/directx/dml-stable-diffusion/)

## Get Started and Resources
- Documentation: [https://microsoft.github.io/Olive](https://microsoft.github.io/Olive)
- Examples: [examples](./examples)

## Installation
We recommend installing Olive in a [virtual environment](https://docs.python.org/3/library/venv.html) or a
[conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Olive is installed using
pip.

Create a virtual/conda environment with the desired version of Python and activate it.

You will need to install a build of [**onnxruntime**](https://onnxruntime.ai). You can install the desired build separately but
public versions of onnxruntime can also be installed as extra dependencies during Olive installation.

### Install with pip
Olive is available for installation from PyPI.
```
pip install olive-ai
```
With onnxruntime (Default CPU):
```
pip install olive-ai[cpu]
```
With onnxruntime-gpu:
```
pip install olive-ai[gpu]
```
With onnxruntime-directml:
```
pip install olive-ai[directml]
```

### Optional Dependencies
Olive has optional dependencies that can be installed to enable additional features. Please refer to [extra dependencies](./olive/extra_dependencies.json) for
the list of extras and their dependencies.

## Pipeline Status

[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive%20CI?label=Olive-CI)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1240)

[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive-ORT-stable?label=Olive-ORT-stable)](https://aiinfra.visualstudio.com/PublicPackages/_build?definitionId=1281)

[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive-ORT-Nightly?label=Olive-ORT-Nightly)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1279)

## Contributing
We’d love to embrace your contribution to Olive. Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).


## License
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.
