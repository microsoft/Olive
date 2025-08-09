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

### ✅ Benefits of using Olive

- **Reduce frustration** of manual trial-and-error model optimization experimentation. Define your target and precision and let Olive automatically produce the best model for you.
- **40+ built-in model optimization components** covering industry-leading techniques across model compression, optimization, finetuning, and compilation.
- **Easy-to-use CLI** for common model optimization tasks.
- **Workflows** to orchestrate model transformations and optimizations steps.
- Support for compiling LoRA adapters for **MultiLoRA serving**.
- Seamless integration with **Hugging Face** and **Azure AI**.
- Built-in **caching** mechanism to **improve productivity**.


## 📰 News Highlights
Here are some recent videos, blog articles and labs that highlight Olive:

- [ Feb 2025 ] [New Notebook available - Finetune and Optimize DeepSeek R1 with Olive 🐋 ](examples/getting_started/olive-deepseek-finetune.ipynb)
- [ Nov 2024 ] [Democratizing AI Model optimization with the new Olive CLI](https://onnxruntime.ai/blogs/olive-cli)
- [ Nov 2024 ] [Unlocking NLP Potential: Fine-Tuning with Microsoft Olive (Ignite Pre-Day Lab PRE016)](https://github.com/Azure/Ignite_FineTuning_workshop)
- [ Nov 2024 ] [Olive supports generating models for MultiLoRA serving on the ONNX Runtime ](https://onnxruntime.ai/blogs/multilora)
- [ Oct 2024 ] [Windows Dev Chat: Optimizing models from Hugging Face for the ONNX Runtime (video)](https://www.youtube.com/live/lAc1fq_0ftw?t=775s)
- [ May 2024 ] [AI Toolkit - VS Code Extension that uses Olive to fine tune models](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio)

For a full list of news and blogs, read the [news archive](./NEWS.md).

## 🚀 Getting Started

### Notebooks available!

The following notebooks are available that demonstrate key optimization workflows with Olive and include the application code to inference the optimized models on the ONNX Runtime.

| Title | Task | Description | Time Required |Notebook Links
| -------- | ------------ | ------------ |-------- | -------- |
| **Quickstart** | Text Generation | *Learn how to quantize & optimize an SLM for the ONNX Runtime using a single Olive command.* | 5mins  | [Download](examples/getting_started/olive_quickstart.ipynb) / [Open in Colab](https://colab.research.google.com/github/microsoft/Olive/blob/main/examples/getting_started/olive_quickstart.ipynb) |
| **Optimizing popular SLMs** | Text Generation | *Choose from a curated list of over 20 popular SLMs to quantize & optimize for the ONNX runtime.* | 5mins  | [Download](examples/getting_started/text-gen-optimized-slms.ipynb) / [Open in Colab](https://colab.research.google.com/github/microsoft/Olive/blob/main/examples/getting_started/text-gen-optimized-slms.ipynb) |
| **How to finetune models for on-device inference** | Text Generation | *Learn how to Quantize (using AWQ method), fine-tune, and optimize an SLM for on-device inference.* |15mins| [Download](examples/getting_started/olive-awq-ft-llama.ipynb) / [Open in Colab](https://colab.research.google.com/github/microsoft/Olive/blob/main/examples/getting_started/olive-awq-ft-llama.ipynb) |
| **Finetune and Optimize DeepSeek R1 with Olive** | Text Generation | *Learn how to Finetune and Optimize DeepSeek-R1-Distill-Qwen-1.5B for on-device inference.* |15mins| [Download](examples/getting_started/olive-deepseek-finetune.ipynb) / [Open in Colab](https://colab.research.google.com/github/microsoft/Olive/blob/main/examples/getting_started/olive-deepseek-finetune.ipynb) |

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
- [Examples](./examples)

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
[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive%2FOlive%20Docs?label=Olive-Docs)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=2064)

