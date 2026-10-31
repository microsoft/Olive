<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/images/olive-white-text.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/images/olive-black-text.png">
    <img alt="olive text" src="docs/source/images/olive-black-text.png" height="100" style="max-width: 100%;">
  </picture>

[![PyPI release](https://img.shields.io/pypi/v/olive-ai)](https://pypi.org/project/olive-ai/)
[![Documentation](https://img.shields.io/website/https/microsoft.github.io/Olive?down_color=red&down_message=offline&up_message=online)](https://microsoft.github.io/Olive/)

## 针对 ONNX Runtime 的 AI 模型优化工具包

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>
</div>

给定模型和目标硬件，Olive（**O**nnx **LIVE** 的缩写）能够组合最合适的优化技术，在兼顾精度与延迟等一系列约束条件的同时，输出用于云端或边缘端推理的最高效 ONNX 模型。

## 📰 最新动态
以下是近期聚焦 Olive 的视频、博文与实验工坊：

- [ 2025年10月 ] [使用 Olive 探索小语言模型（SLM）的最佳量化配置](https://microsoft.github.io/Olive/blogs/quant-slms.html)
- [ 2025年9月 ] [Olive 示例已迁移至全新的 Olive-recipes 仓库](https://github.com/microsoft/olive-recipes)
- [ 2025年8月 ] [Olive 0.9.2 正式发布，带来全新量化算法](https://github.com/microsoft/Olive/releases/tag/v0.9.2)
- [ 2025年5月 ] [Olive 0.9.0 正式发布，新增对 NPU 的支持](https://github.com/microsoft/Olive/releases/tag/v0.9.0)
- [ 2025年3月 ] [Olive 0.8.0 正式发布，支持全新量化技术](https://github.com/microsoft/Olive/releases/tag/v0.8.0)
- [ 2025年2月 ] [全新 Notebook 上线 - 使用 Olive 微调与优化 DeepSeek R1 🐋](https://github.com/microsoft/Olive/blob/main/notebooks/olive-deepseek-finetune.ipynb)
- [ 2024年11月 ] [通过全新 Olive CLI 普及 AI 模型优化能力](https://onnxruntime.ai/blogs/olive-cli)
- [ 2024年11月 ] [释放 NLP 潜能：使用 Microsoft Olive 进行模型微调（Ignite Pre-Day Lab PRE016）](https://github.com/Azure/Ignite_FineTuning_workshop)
- [ 2024年11月 ] [Olive 支持生成用于在 ONNX Runtime 上进行 MultiLoRA 服务的模型](https://onnxruntime.ai/blogs/multilora)
- [ 2024年10月 ] [Windows Dev Chat：针对 ONNX Runtime 优化 Hugging Face 模型（视频）](https://www.youtube.com/live/lAc1fq_0ftw?t=775s)
- [ 2024年5月 ] [AI Toolkit - 使用 Olive 进行模型微调的 VS Code 插件](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio)

欲查看完整新闻与博客列表，请阅读[新闻归档](./NEWS.md)。

## 🚀 快速上手

### ✨ 极速入门
如果您更倾向于直接使用命令行而非 Jupyter Notebook，我们在下方整理了快速上手的核心命令。

#### 1. 安装 Olive CLI
我们建议在 [虚拟环境（venv）](https://docs.python.org/3/library/venv.html) 或 [Conda 环境](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 中安装 Olive。

```
pip install olive-ai
pip install transformers onnxruntime-genai
```
> [!NOTE]
> Olive 包含可选依赖项，安装后可启用更多功能。关于扩展（extras）及其依赖列表，请参阅 [Olive 安装包配置](./olive/olive_config.json)。

> [!NOTE]
> 针对 Windows 用户：如何避免 HF_HUB_DISABLE_SYMLINKS_WARNING 警告
>
> 如果您从 Hugging Face 下载模型，Olive 将依赖 huggingface_hub 库。在 Windows 平台上，您可能会收到如下警告：
>
> ```
> UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\name\.cache\huggingface\hub\model-name.
> Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
> ```
>
> 您可以通过以下 4 种方案之一来解决此警告。请注意，方案 1、2、3 可以节约磁盘空间，而方案 4 仅抑制警告提示。
>
> 1. 在 Windows 上启用开发者模式（一次性设置），详见 [Microsoft 开发者工具文档](https://learn.microsoft.com/en-us/windows/advanced-settings/developer-mode)。
> 2. 在将 Olive 与 huggingface_hub 库配合使用时以管理员身份运行 Python。
> 3. 重新配置文件缓存存储路径。该限制不仅限于在默认位置存储缓存。
> 4. 继续设置 `HF_HUB_DISABLE_SYMLINKS_WARNING=1` 以抑制警告，并接受额外的磁盘空间占用。
>
> 针对上述方案，您可以根据自身环境所允许的条件（例如企业安全策略）选择最适合您的方式。
>
> huggingface_hub 库的这一限制在 [Hub 客户端库文档](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#limitations) 中也有详细说明。

#### 2. 自动优化器

在本次快速入门中，您将对 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 进行优化。该模型在 Hugging Face 仓库中包含针对不同精度的多个模型文件，而 Olive 并不需要全部文件。

运行自动优化流程：

```bash
olive optimize \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --precision int4 \
    --output_path models/qwen
```

>[!TIP]
><details>
><summary>PowerShell 用户提示</summary>
>Bash 与 PowerShell 之间的行连接符不可互换。如果您使用的是 PowerShell，可以复制并粘贴以下使用兼容行连接符的命令：
>
>```powershell
>olive optimize `
>    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct `
>    --output_path models/qwen `
>    --precision int4
>```
></details>
><br>

自动优化器将执行以下步骤：

1. 从 Hugging Face 模型仓库下载获取模型。
1. 使用 GPTQ 将模型量化为 `int4` 精度。
1. 捕获 ONNX 计算图，并将权重存储在 ONNX 数据文件中。
1. 深度优化 ONNX 计算图。

Olive 开箱即可自动优化主流模型**架构**（如 Llama、Phi、Qwen、Gemma 等）——[查看详细支持列表](https://huggingface.co/docs/optimum/en/exporters/onnx/overview)。此外，您也可以通过提供模型输入/输出配置（`io_config`）来优化其他自定义模型架构。

#### 3. 在 ONNX Runtime 上进行推理

ONNX Runtime (ORT) 是一款高速、轻量级的跨平台推理引擎，提供了针对 Python、C/C++、C#、Java、JavaScript 等主流编程语言的 API 绑定。ORT 能够帮助您将 AI 模型轻松注入各类应用程序中，实现端侧/设备侧高效推理。

可运行的聊天对话示例程序位于 [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) GitHub 仓库中的 [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py)。

## 🎓 了解更多

- [官方文档](https://microsoft.github.io/Olive)
- [配方库 (Recipes)](https://github.com/microsoft/olive-recipes)

## 数据与遥测 (Data/Telemetry)
本项目的发行版本可能会收集使用数据并发送给 Microsoft，以帮助改进我们的产品和服务。详情请参阅[隐私声明](docs/Privacy.md)。

## 🤝 贡献与反馈
- 欢迎社区贡献！请阅读[贡献指南](./CONTRIBUTING.md)了解参与 Olive 项目贡献的详细流程。
- 如需提交功能建议或报告 Bug，请提交 [GitHub Issue](https://github.com/microsoft/Olive/issues)。
- 参与常规讨论或提问，请前往 [GitHub Discussions](https://github.com/microsoft/Olive/discussions)。

## ⚖️ 开源许可证
Copyright (c) Microsoft Corporation. All rights reserved.

根据 [MIT](./LICENSE) 许可证授权。

## 流水线状态 (Pipeline Status)

[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive%20CI?label=Olive-CI)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1240)
[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive-ORT-Nightly?label=Olive-ORT-Nightly)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1279)

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月31日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
