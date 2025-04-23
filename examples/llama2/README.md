
- [Prerequisites](#prerequisites)
  - [Clone the repository and install Olive](#clone-the-repository-and-install-olive)
  - [Install onnxruntime](#install-onnxruntime)
  - [Install extra dependencies](#install-extra-dependencies)

- [Llama2 optimization](#llama2-optimization)
    - [CPU](#cpu)
    - [GPU](#gpu)
    - [DirectML](#directml)
    - [QLoRA finetune](#qlora-finetune)
    - [Running optimized models](#running-optimized-models)
      - [Inference using ONNX Runtime GenAI](#inference-using-onnx-runtime-genai)
      - [Inference using Optimum](#inference-using-optimum)
    - [Workflows](#workflows)
      - [Running Workflows on the cloud](#running-workflows-on-the-cloud)
      - [Accelerating Workflows with shared cache](#accelerating-workflows-with-shared-cache)
      - [Combining Remote Workflow and shared cache](#combining-remote-workflow-and-shared-cache)
- [License](#license)

# Prerequisites

## Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

## Install onnxruntime

This example requires onnxruntime>=1.16.2. Please install the latest version of onnxruntime:

For CPU:

```bash
python -m pip install "onnxruntime>=1.17.0"
```

For GPU:

```bash
python -m pip install "onnxruntime-gpu>=1.17.0"
```

**Note:** The GPU package also works for CPU.

## Install extra dependencies

Install the necessary python packages:
```
python -m pip install -r <requirements_file>.txt
```

# Llama2 optimization

Sample use cases of Olive to optimize a [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf)

## CPU

- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32 -> Onnx Dynamic Quantization*
- CPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32 -> Onnx Block wise int4 Quantization*

Requirements file: [requirements.txt](requirements.txt)

```bash
# Optimize the model: FP32/INT8/INT4
python llama2.py --model_name meta-llama/Llama-2-7b-hf
```

## GPU

- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 + Grouped Query Attention (optional)*
- GPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 + Grouped Query Attention (optional) -> Onnx Block wise int4 Quantization*
- GPU, GPTQ INT4: *PyTorch Model -> GPTQ INT4 Onnx Model*

**Note:** Group Query Attention is optional and can be enabled by passing `--use_gqa` flag to the script. It is only supported for GPU.

Requirements file: [requirements.txt](requirements.txt)
Requirements file: [requirements-gptq.txt](requirements-gptq.txt)

```bash
# Optimize the model: FP16/INT4
python llama2.py --model_name meta-llama/Llama-2-7b-hf --gpu
# use gqa instead of mha
python llama2.py --model_name meta-llama/Llama-2-7b-hf --gpu --use_gqa
# use gptq quantization
python llama2.py --model_name meta-llama/Llama-2-7b-hf --gpu --use_gptq
```

## DirectML

For Llama2 inference with DirectML on GPUs, pls refer to this [example](https://github.com/microsoft/Olive/tree/main/examples/directml/llm).

## QLoRA finetune

You can finetune llama2 on a code generation dataset using QLoRA and produce optimized ONNX model by running the following command:

```bash
python llama2.py --qlora
```

## Running optimized models

### Inference using ONNX Runtime GenAI

For using ONNX runtime GenAI to optimize, follow build and installation instructions [here](https://github.com/microsoft/onnxruntime-genai) to install onnxruntime-genai package(>0.1.0).

Run the following command to execute the workflow:

```bash
python llama2_model_builder.py [--model_name <>] [--metadata_only]
```

To generate metadata only for pre-exported onnx model, use the `--metadata_only` option.

Snippet below shows an example run of generated llama2 model.

```python
import onnxruntime_genai as og

model = og.Model("model_path")
tokenizer = og.Tokenizer(model)

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=200)
generator = og.Generator(model, params)
generator.append_tokens(tokens)

while not generator.is_done():
  generator.generate_next_token()

text = tokenizer.decode(generator.get_sequence(0))

print("Output:")
print(text)
```

### Inference using Optimum

You can do text generation using the following code:

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, AutoConfig

quantized_model_dir = "${path_to_quantized_llama2-7b}"
AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf").save_pretrained(quantized_model_dir)
AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf").save_pretrained(quantized_model_dir)
model = ORTModelForCausalLM.from_pretrained(
    quantized_model_dir, provider="CUDAExecutionProvider"
)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
inputs = tokenizer("Hello, World", return_tensors="pt").to("cuda:0")
print(tokenizer.batch_decode(model.generate(**inputs, max_length=20), skip_special_tokens=True))
```

## Workflows

### Running Workflows on the Cloud

You may notice that this workflow takes a long time to run, especially for QLoRA. Olive offers a feature that allows you to submit the workflow to the cloud, enabling it to run on the compute resources in your Azure Machine Learning workspace.

To use this feature, you will need a `remote_config.json` file to configure your Azure Machine Learning workspace:

```json
{
    "subscription_id": "<subscription_id>",
    "resource_group": "<resource_group>",
    "workspace_name": "<workspace_name>",
    "keyvault_name": "<keyvault_name>",
    "compute": "<compute>"
}
```

More details about `keyvault_name` can be found [here](https://microsoft.github.io/Olive/features/huggingface-integration.html#huggingface-login).

Make sure you have installed Olive Azure ML extra by running:

```bash
pip install olive-ai[azureml]
```

Then you can run the following command:

```bash
python llama2.py --qlora --remote_config remote_config.json
```

Olive will submit the workflow to the compute resources in your Azure Machine Learning workspace and execute the workflow there. The output artifacts will be automatically exported to the Datastore. For more detailed information, please refer to [the official documentation](https://microsoft.github.io/Olive/features/azure-ai/remote-workflow.html).

### Accelerating Workflows with shared cache

The shared cache is a system where Olive stores intermediate models in Azure Blob Storage. For more detailed information, please refer to [the documentation](https://microsoft.github.io/Olive/features/azure-ai/shared-model-cache.html).

You can run the following command:

```bash
python llama2.py --qlora --account_name <account_name> --container_name <container_name>
```

Olive will apply shared model cache for this workflow.

### Combining Remote Workflow and Shared Cache

To leverage both the remote workflow and shared cache for faster workflow execution, simply run:

```bash
python llama2.py --qlora --remote_config remote_config.json --account_name <account_name> --container_name <container_name>
```

This will submit the workflow to the Azure Machine Learning workspace and store intermediate models in Azure Blob Storage, significantly speeding up the process.

# License
Please see the [LICENSE](./LICENSE) file for more details. Also please follow the [user policy](./USE-POLICY-META-LLAMA-2.md) of the model provider. Besides, please refer to the [Responsible
Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/) for more details on how to use the model responsibly.
