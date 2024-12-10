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

Given a model and targeted hardware, Olive (**O**nnx **Live**) composes the best suitable optimization techniques to output the most efficient ONNX model(s) for inferring on cloud or edge, while taking a set of constraints such as accuracy and latency into consideration. 

### ➕ Benefits of using Olive

- **Reduce frustration** of manual trial-and-error model optimization experimentation. Define your target and precision and let Olive automatically produce the best model for you.
- **40+ built-in model optimization components** covering industry-leading techniques across model compression, optimization, finetuning, and compilation.
- **Easy-to-use CLI** for common model optimization tasks.
- **Workflows** to orchestrate model transformations and optimizations steps.
- Support for compiling LoRA adapters for **MultiLoRA serving**.
- Seamless integration with **Hugging Face** and **Azure AI**.
- Built-in **caching** mechanism to **improve productivity**.


## 📰 News Highlights
Here are some recent videos, blog articles and labs that highlight Olive:

- [ Nov 2024 ] [Democratizing AI Model optimization with the new Olive CLI](https://onnxruntime.ai/blogs/olive-cli)
- [ Nov 2024 ] [Unlocking NLP Potential: Fine-Tuning with Microsoft Olive (Ignite Pre-Day Lab PRE016)](https://github.com/Azure/Ignite_FineTuning_workshop)
- [ Nov 2024 ] [Olive supports generating models for MultiLoRA serving on the ONNX Runtime ](https://onnxruntime.ai/blogs/multilora)
- [ Oct 2024 ] [Windows Dev Chat: Optimizing models from Hugging Face for the ONNX Runtime (video)](https://www.youtube.com/live/lAc1fq_0ftw?t=775s)
- [ May 2024 ] [AI Toolkit - VS Code Extension that uses Olive to fine tune models](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio)

For a full list of news and blogs, read the [news archive](./NEWS.md).

## 🚀 Getting Started

### 📓 Notebooks available!

The following notebooks are available that demonstrate key optimization workflows with Olive:

| Title | Description | Time Required |Notebook Links 
| -------- | ------------ | -------- | -------- 
| Quickstart | *In this notebook you will use Olive's automatic optimizer to ONNX Runtime on a CPU Device and then inference the model using the ONNX Runtime Generate API* | 5mins  | [Download](examples/getting_started/olive_quickstart.ipynb)<br>[Open in Colab](https://colab.research.google.com/github/microsoft/Olive/blob/main/examples/getting_started/olive_quickstart.ipynb) |
| Quantize and Finetune | *In this notebook you will (1) quantize Llama-3.2-1B-Instruct using the AWQ algorithm, (2) fine-tune the quantized model, (3) Optimize the fine-tuned model for the ONNX Runtime, and (4) Inference the fine-tuned model using the ONNX runtime Generate API.* |15mins| [Download](examples/getting_started/olive-awq-ft-llama.ipynb)<br>[Open in Colab](https://colab.research.google.com/github/microsoft/Olive/blob/main/examples/getting_started/olive-awq-ft-llama.ipynb) |

### ✨ Quickstart
We recommend the [quickstart notebook](examples/getting_started/olive_quickstart.ipynb), however if you prefer not to use Jupyter notebooks then you can run through the following steps.

#### 💾 1. Install Olive CLI
We recommend installing Olive in a [virtual environment](https://docs.python.org/3/library/venv.html) or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

```
pip install olive-ai[ort-genai,auto-opt]
pip install transformers==4.44.2
```
> [!NOTE]
> Olive has optional dependencies that can be installed to enable additional features. Please refer to [Olive package config](./olive/olive_config.json) for the list of extras and their dependencies.

#### 🪄 2. Automatic Optimizer

You'll be optimizing the [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) model for CPU devices with the `auto-opt` command. To minimize the download you can cache just the safetensors and configuration files from the Hugging Face repo using:

```bash
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct *.json *.safetensors *.txt
```

Next, run the automatic optimization using:

```bash
olive auto-opt \
    --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct \
    --output_path models/smolm2 \
    --device cpu \
    --provider CPUExecutionProvider \
    --use_ort_genai \
    --precision int4 \
    --log_level 1
```

The automatic optimizer will:

1. Acquire the model from either Hugging Face or the local cache.
1. Capture the ONNX Graph and store the weights in an ONNX data file.
1. Optimize the ONNX Graph.
1. Quantize the model to `int4` using RTN method.

> [!NOTE]
> Olive can automatically optimize popular model *architectures* like Llama, Phi, Qwen, Gemma, etc out-of-the-box - [see detailed list here](https://huggingface.co/docs/optimum/en/exporters/onnx/overview). Also, you can optimize other model architectures by providing details on the input/outputs of the model (`io_config`).


#### 🧠 3. Inference on the ONNX Runtime

The ONNX Runtime (ORT) is a fast and light-weight package (available with many programming language bindings such as Python, C/C++, C#, Java, JavaScript, etc) that runs cross-platform. ORT enables you to infuse your AI models into your applications so that inference is handled on-device. The following code creates a simple console-based chat interface that inferences your optimized model.

You'll be prompted to enter a message to the SLM - for example, you could ask *what is the golden ratio*, or *def print_hello_world():*. To exit type *exit* in the chat interface.

```python
import onnxruntime_genai as og

model_folder = "models/smolm2/model"

# Load the base model and tokenizer
model = og.Model(model_folder)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options['max_length'] = 200
search_options['past_present_share_buffer'] = False

chat_template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"

text = input("Input: ")

# Keep asking for input phrases
while text != "exit":
    if not text:
        print("Error, input cannot be empty")
        exit

    # generate prompt (prompt template + input)
    prompt = f'{chat_template.format(input=text)}'

    # encode the prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    params.input_ids = input_tokens
    generator = og.Generator(model, params)

    print("Output: ", end='', flush=True)
    # stream the output
    try:
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end='', flush=True)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")

    print()
    text = input("Input: ")
```

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
[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive%20AzureML%20Example%20Test?label=Olive-AML-CI)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1541)
[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive-ORT-stable?label=Olive-ORT-stable)](https://aiinfra.visualstudio.com/PublicPackages/_build?definitionId=1281)
[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive-ORT-Nightly?label=Olive-ORT-Nightly)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1279)

