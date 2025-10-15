<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/images/olive-white-text.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/images/olive-black-text.png">
    <img alt="olive text" src="docs/source/images/olive-black-text.png" height="100" style="max-width: 100%;">
  </picture>

[![PyPI release](https://img.shields.io/pypi/v/olive-ai)](https://pypi.org/project/olive-ai/)
[![Documentation](https://img.shields.io/website/https/microsoft.github.io/Olive?down_color=red&down_message=offline&up_message=online)](https://microsoft.github.io/Olive/)

## AI Model Optimization Toolkit for the ONNX Runtime
</div>

Given a model and targeted hardware, Olive (abbreviation of **O**nnx **LIVE**) composes the best suitable optimization techniques to output the most efficient ONNX model(s) for inferencing on the cloud or edge, while taking a set of constraints such as accuracy and latency into consideration.

## 📰 News Highlights
Here are some recent videos, blog articles and labs that highlight Olive:

- [ Oct 2025 ] [Exploring Optimal Quantization Settings for Small Language Models with Olive](https://microsoft.github.io/Olive/blogs/quant-slms.html)
- [ Sep 2025 ] [Olive examples are relocated to new Olive-recipes repository](https://github.com/microsoft/olive-recipes)
- [ Aug 2025 ] [Olive 0.9.2 is released with new quantization algorithms](https://github.com/microsoft/Olive/releases/tag/v0.9.2)
- [ May 2025 ] [Olive 0.9.0 is released with support for NPUs](https://github.com/microsoft/Olive/releases/tag/v0.9.0)
- [ Mar 2025 ] [Olive 0.8.0 is released with new quantization techniques](https://github.com/microsoft/Olive/releases/tag/v0.8.0)
- [ Feb 2025 ] [New Notebook available - Finetune and Optimize DeepSeek R1 with Olive 🐋 ](https://github.com/microsoft/Olive/blob/main/notebooks/olive-deepseek-finetune.ipynb)
- [ Nov 2024 ] [Democratizing AI Model optimization with the new Olive CLI](https://onnxruntime.ai/blogs/olive-cli)
- [ Nov 2024 ] [Unlocking NLP Potential: Fine-Tuning with Microsoft Olive (Ignite Pre-Day Lab PRE016)](https://github.com/Azure/Ignite_FineTuning_workshop)
- [ Nov 2024 ] [Olive supports generating models for MultiLoRA serving on the ONNX Runtime ](https://onnxruntime.ai/blogs/multilora)
- [ Oct 2024 ] [Windows Dev Chat: Optimizing models from Hugging Face for the ONNX Runtime (video)](https://www.youtube.com/live/lAc1fq_0ftw?t=775s)
- [ May 2024 ] [AI Toolkit - VS Code Extension that uses Olive to fine tune models](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio)

For a full list of news and blogs, read the [news archive](./NEWS.md).

## 🚀 Getting Started

### ✨ Quickstart
If you prefer using the command line directly instead of Jupyter notebooks, we've outlined the quickstart commands here.

#### 1. Install Olive CLI
We recommend installing Olive in a [virtual environment](https://docs.python.org/3/library/venv.html) or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```
pip install olive-ai[auto-opt]
pip install transformers onnxruntime-genai
```
> [!NOTE]
> Olive has optional dependencies that can be installed to enable additional features. Please refer to [Olive package config](./olive/olive_config.json) for the list of extras and their dependencies.

#### 2. Automatic Optimizer

In this quickstart you'll be optimizing [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), which has many model files in the Hugging Face repo for different precisions that are not required by Olive.

Run the automatic optimization:

```bash
olive optimize \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --precision int4 \
    --output_path models/qwen
```

>[!TIP]
><details>
><summary>PowerShell Users</summary>
>Line continuation between Bash and PowerShell are not interchangable. If you are using PowerShell, then you can copy-and-paste the following command that uses compatible line continuation.
>
>```powershell
>olive optimize `
>    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct `
>    --output_path models/qwen `
>    --precision int4
>```
</details>
<br>

The automatic optimizer will:

1. Acquire the model from the the Hugging Face model repo.
1. Quantize the model to `int4` using GPTQ.
1. Capture the ONNX Graph and store the weights in an ONNX data file.
1. Optimize the ONNX Graph.

Olive can automatically optimize popular model *architectures* like Llama, Phi, Qwen, Gemma, etc out-of-the-box - [see detailed list here](https://huggingface.co/docs/optimum/en/exporters/onnx/overview). Also, you can optimize other model architectures by providing details on the input/outputs of the model (`io_config`).


#### 3. Inference on the ONNX Runtime

The ONNX Runtime (ORT) is a fast and light-weight cross-platform inference engine with bindings for popular programming language such as Python, C/C++, C#, Java, JavaScript, etc. ORT enables you to infuse AI models into your applications so that inference is handled on-device.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

## 🎓 Learn more

- [Documentation](https://microsoft.github.io/Olive)
- [Recipes](https//github.com/microsoft/olive-recipes)

## 🤝 Contributions and Feedback
- We welcome contributions! Please read the [contribution guidelines](./CONTRIBUTING.md) for more details on how to contribute to the Olive project.
- For feature requests or bug reports, file a [GitHub Issue](https://github.com/microsoft/Olive/issues).
- For general discussion or questions, use [GitHub Discussions](https://github.com/microsoft/Olive/discussions).


## ⚖️ License
Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.

## Pipeline Status

[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive%20CI?label=Olive-CI)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1240)
[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive-ORT-Nightly?label=Olive-ORT-Nightly)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1279)
