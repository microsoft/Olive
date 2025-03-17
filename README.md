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
pip install transformers==4.44.2 onnxruntime-genai
```
> [!NOTE]
> Olive has optional dependencies that can be installed to enable additional features. Please refer to [Olive package config](./olive/olive_config.json) for the list of extras and their dependencies.

#### 2. Automatic Optimizer

In this quickstart you'll be optimizing [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), which has many model files in the Hugging Face repo for different precisions that are not required by Olive. To minimize the download, cache the original Hugging Face model files (safetensors and configuration) in the main folder of the Hugging Face repo using:

```bash
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct *.json *.safetensors *.txt
```

Next, run the automatic optimization:

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

>[!TIP]
><details>
><summary>PowerShell Users</summary>
>Line continuation between Bash and PowerShell are not interchangable. If you are using PowerShell, then you can copy-and-paste the following command that uses compatible line continuation.
>
>```powershell
>olive auto-opt `
>    --model_name_or_path HuggingFaceTB/SmolLM2-135M-Instruct `
>    --output_path models/smolm2 `
>    --device cpu `
>    --provider CPUExecutionProvider `
>    --use_ort_genai `
>    --precision int4 `
>    --log_level 1
>```
</details>
<br>

The automatic optimizer will:

1. Acquire the model from the local cache (note: if you skipped the model download step then the entire contents of the Hugging Face model repo will be downloaded).
1. Capture the ONNX Graph and store the weights in an ONNX data file.
1. Optimize the ONNX Graph.
1. Quantize the model to `int4` using RTN method.

Olive can automatically optimize popular model *architectures* like Llama, Phi, Qwen, Gemma, etc out-of-the-box - [see detailed list here](https://huggingface.co/docs/optimum/en/exporters/onnx/overview). Also, you can optimize other model architectures by providing details on the input/outputs of the model (`io_config`).


#### 3. Inference on the ONNX Runtime

The ONNX Runtime (ORT) is a fast and light-weight cross-platform inference engine with bindings for popular programming language such as Python, C/C++, C#, Java, JavaScript, etc. ORT enables you to infuse AI models into your applications so that inference is handled on-device.

The following code creates a simple console-based chat interface that inferences your optimized model - **select Python and/or C# to expand the code:**

<details>
<summary><b>Python</b></summary

Create a Python file called `app.py` and copy and paste the following code:
```python
# app.py
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

chat_template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"

# Keep asking for input prompts in a loop
while True:
    text = input("Prompt (Use quit() to exit): ")
    if not text:
        print("Error, input cannot be empty")
        continue

    if text == "quit()":
        break

    # Generate prompt (prompt template + input)
    prompt = f'{chat_template.format(input=text)}'

    # Encode the prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt)

    # Create params and generator
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)

    # Append input tokens to the generator
    generator.append_tokens(input_tokens)

    print("")
    print("Output: ", end='', flush=True)
    # Stream the output
    try:
        while not generator.is_done():
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end='', flush=True)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")
    print()
    print()

    del generator
```
To run the code, execute `python app.py`. You'll be prompted to enter a message to the SLM - for example, you could ask *what is the golden ratio*, or *def print_hello_world():*. To exit type *quit()* in the chat interface.

</details>

<details>
<summary><b>C#</b></summary>

Create a new C# Console app and install the [Microsoft.ML.OnnxRuntimeGenAI](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI) Nuget package into your project:

```powershell
mkdir ortapp
cd ortapp
dotnet new console
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.5.2
```

Next, copy-and-paste the following code into your `Program.cs` file and update `modelPath` variable to be the *absolute path* of where you stored your optimized model.

```csharp
// Program.cs
using Microsoft.ML.OnnxRuntimeGenAI;

internal class Program
{
    private static void Main(string[] args)
    {
        string modelPath @"models/smolm2/model";

        Console.Write("Loading model from " + modelPath + "...");
        using Model model = new(modelPath);
        Console.Write("Done\n");
        using Tokenizer tokenizer = new(model);
        using TokenizerStream tokenizerStream = tokenizer.CreateStream();


        while (true)
        {
            Console.Write("User:");

            string prompt = "<|im_start|>user\n" +
                            Console.ReadLine() +
                            "<|im_end|>\n<|im_start|>assistant\n";
            var sequences = tokenizer.Encode(prompt);

            using GeneratorParams gParams = new GeneratorParams(model);
            gParams.SetSearchOption("max_length", 200);
            using Generator generator = new(model, gParams);
            generator.AppendTokenSequences(sequences);

            Console.Out.Write("\nAI:");
            while (!generator.IsDone())
            {
                generator.GenerateNextToken();
                var token = generator.GetSequence(0)[^1]
                Console.Out.Write(tokenizerStream.Decode(token));
                Console.Out.Flush();
            }
            Console.WriteLine();
        }
    }
}
```

Run the application:

```powershell
dotnet run
```

You'll be prompted to enter a message to the SLM - for example, you could ask *what is the golden ratio*, or *def print_hello_world():*. To exit type *exit* in the chat interface.

</details>

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
[![Build Status](https://dev.azure.com/aiinfra/PublicPackages/_apis/build/status%2FOlive-ORT-Nightly?label=Olive-ORT-Nightly)](https://dev.azure.com/aiinfra/PublicPackages/_build/latest?definitionId=1279)

