# Llama2 optimization
Sample use cases of Olive to optimize a [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf)

- [Inference Optimization with ONNX Runtime tools for CPUs and GPUs](#inference-optimize-using-onnx-runtime-tools)
- [Inference Optimization with ONNX Runtime DirectML for GPUs](#inference-optimization-with-onnnx-runtime-with-directml)
- [With QLoRa fine-tune and ONNX Runtime Inference Optimizations](#fine-tune-on-a-code-generation-dataset-using-qlora-and-optimize-using-onnx-runtime-tools)
- [Notebook of using AzureML compute to fine tune and optimize for your local GPUs](https://github.com/microsoft/Olive/tree/main/examples/llama2/notebook)
- [How to run](#prerequisites)

## Optimization Workflows
### Inference optimization using ONNX Runtime Tools
Performs optimization pipeline:
- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32 -> Onnx Dynamic Quantization*
- CPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32 -> Onnx Block wise int4 Quantization*
- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 + Grouped Query Attention (optional)*
- GPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 + Grouped Query Attention (optional) -> Onnx Block wise int4 Quantization*

**Note:** Group Query Attention is optional and can be enabled by passing `--use_gqa` flag to the script. It is only supported for GPU.

Requirements file: [requirements.txt](requirements.txt)

### Inference optimization with ONNNX Runtime with DirectML
For Llama2 inference with DirectML on GPUs, pls refer to this [example](https://github.com/microsoft/Olive/tree/main/examples/directml/llama_v2).

### Fine-tune on a code generation dataset using QLoRA and optimize using ONNX Runtime Tools
This workflow fine-tunes Open LLaMA model using [QLoRA](https://arxiv.org/abs/2305.14314) to generate code given a prompt. The fine-tuned model is then optimized using ONNX Runtime Tools.
Performs optimization pipeline:
- GPU, NF4: *Pytorch Model -> Fine-tuned Pytorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 -> Onnx Bitsandbytes 4bit Quantization*

**Note:**
- This workflow is only supported for GPU.
- The relevant config file is [llama2_qlora.json](llama2_qlora.json). The code language is set to `Python` but can be changed to other languages by changing the `language` field in the config file.
Supported languages are Python, TypeScript, JavaScript, Ruby, Julia, Rust, C++, Bash, Java, C#, and Go. Refer to the [dataset card](https://huggingface.co/datasets/nampdn-ai/tiny-codes) for more details on the dataset.
- You must be logged in to HuggingFace using `huggingface-cli login` to download the dataset or update `token` field in the config file with your HuggingFace token.

Requirements file: [requirements-qlora.txt](requirements-qlora.txt)

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Install onnxruntime
This example requires onnxruntime>=1.16.2. Please install the latest version of onnxruntime:

For CPU:
```bash
python -m pip install "onnxruntime>=1.16.2"
```

For GPU:
```bash
python -m pip install "onnxruntime-gpu>=1.16.2"
```

**Note:** The GPU package also works for CPU.

### Install extra dependencies
Install the necessary python packages:
```
python -m pip install -r <requirements_file>.txt
```

## Run the config to optimize the model
### Optimize using ONNX Runtime Tools
You can only generate the optimized config file by running the following command for double checking before running the optimization pipeline:
```bash
python llama2.py --model_name meta-llama/Llama-2-7b-hf --only_config
```

Or you can run the following command to directly optimize the model:

CPU:
```bash
# run to optimize the model: FP32/INT8/INT4
python llama2.py --model_name meta-llama/Llama-2-7b-hf
```

GPU:
```bash
# run to optimize the model: FP16/INT4
python llama2.py --model_name meta-llama/Llama-2-7b-hf --gpu
# use gqa instead of mha
python llama2.py --model_name meta-llama/Llama-2-7b-hf --gpu --use_gqa
```

### Fine-tune on a code generation dataset using QLoRA and optimize using ONNX Runtime Tools
Run the following command to execute the workflow:
```bash
python -m olive.workflows.run --config lamma2_qlora.json
```

# License
Please see the [LICENSE](./LICENSE) file for more details. Also please follow the [user policy](./USE-POLICY-META-LLAMA-2.md) of the model provider. Besides, please refer to the [Responsible
Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/) for more details on how to use the model responsibly.
