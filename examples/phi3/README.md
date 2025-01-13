# Phi3 optimization with Olive
This folder contains an example of optimizing the Phi-3-Mini-4K-Instruct model from [Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) or [Azure Machine Learning Model Catalog](https://ai.azure.com/explore/models/Phi-3-mini-4k-instruct/version/7/registry/azureml?tid=72f988bf-86f1-41af-91ab-2d7cd011db47) for different hardware targets with Olive.


## Prerequisites
Install the dependencies
```
pip install -r requirements.txt
```
* einops
* Pytorch: >=2.2.0 \
  _The [official website](https://pytorch.org/) offers packages compatible with CUDA 11.8 and 12.1. Please select the appropriate version according to your needs._
* [Package onnxruntime](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages): >=1.18.0
* [Package onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai): >=0.2.0.

If you target GPU, pls install onnxruntime and onnxruntime-genai gpu packages.

<!-- TODO(anyone): Remove this when genai doesn't require login -->
### For optimizing model from Hugging Face
if you have not logged in Hugging Face account,
- Install Hugging Face CLI and login your Hugging Face account for model access
```
huggingface-cli login
```

### For optimizing model from Azure Machine Learning Model Catalog

- Install Olive with Azure Machine Learining dependency
```
pip install olive-ai[azureml]
```
if you have not logged in Azure account,
- Install Azure Command-Line Interface (CLI) following [this link](https://learn.microsoft.com/en-us/cli/azure/)
- Run `az login` to login your Azure account to allows Olive to access the model.

# Usage with CLI
You can use Olive CLI command to export, fine-tune, and optimize the model for a chosen hardware target. Few examples below:

```
# To auto-optimize the exported model
olive auto-opt -m microsoft/Phi-3-mini-4k-instruct --precision int8

# To quantize the model
olive quantize -m microsoft/Phi-3-mini-4k-instruct --implementation gptq

# To tune ONNX session params
olive tune-session-params -m microsoft/Phi-3-mini-4k-instruct --io_bind --enable_cuda_graph
```

For more information on available options to individual CLI command run `olive <command-name> --help` on the command line.

## Usage with custom configuration
we will use the `phi3.py` script to fine-tune and optimize model for a chosen hardware target by running the following commands.

```
python phi3.py [--target HARDWARE_TARGET] [--precision DATA_TYPE] [--source SOURCE] [--finetune_method METHOD] [--inference] [--prompt PROMPT] [--max_length LENGTH]

# Examples
python phi3.py --target mobile

python phi3.py --target mobile --source AzureML

python phi3.py --target mobile --inference --prompt "Write a story starting with once upon a time" --max_length 200

# Fine-tune the model with lora method, optimize the model for cuda target and inference with ONNX Runtime Generate() API
python phi3.py --target cuda --finetune_method lora --inference --prompt "Write a story starting with once upon a time" --max_length 200

# Fine-tune, quantize using AWQ and optimize the model for cpu target
python phi3.py --target cpu --precision int4 --finetune_method lora --awq

# Search and generate an optimized ONNX session tuning config
python phi3.py --target cuda --precision fp16 --tune-session-params
```

Run the following to get more information about the script:
```
python phi3.py --help
```

This script includes
- Generate the Olive configuration file for the chosen HW target
- (optional) Fine-tune model by lora or qlora method with dataset of `nampdn-ai/tiny-codes`.
- (optional) Quantize the original or fine-tuned model using AWQ. If AWQ is not used, the model will be quantized using RTN if precision is int4.
- Generate optimized onnx model with Olive based on the configuration file for the chosen HW target
- Search and generate optimized ONNX session tuning config
- (optional) Inference the optimized model with ONNX Runtime Generate() API with non-web target


If you have an Olive configuration file, you can also run the olive command for model generation:
```
olive run [--config CONFIGURATION_FILE]

# Examples
olive run --config phi3_run_mobile_int4.json
```

### Get access to fine-tuning dataset
Get access to the following resources on Hugging Face Hub:
- [nampdn-ai/tiny-codes](https://huggingface.co/nampdn-ai/tiny-codes)

# Quantize Models with NVIDIA TensorRT Model Optimizer
The **TensorRT Model Optimizer** is designed to bring advanced model compression techniques, including quantization, to Windows RTX PC systems. Engineered for Windows, it delivers rapid and efficient quantization through features such as local GPU calibration, reduced memory usage, and fast processing.
The primary goal of TensorRT Model Optimizer is to produce optimized, ONNX-format models compatible with DirectML backends.
## Setup
Run the following commands to install necessary packages:
```bash
pip install olive-ai[nvmo]
pip install onnxruntime-genai-directml>=0.4.0
pip install onnxruntime-directml==1.20.0
pip install -r requirements-nvmo-awq.txt
```
Install the CUDA version compatible with CuPy as specified in `requirements-nvmo-awq.txt`.
## Validate Installation
After setup, confirm the correct installation of the `modelopt` package by running:
```bash
python -c "from modelopt.onnx.quantization.int4 import quantize as quantize_int4"
```
## Quantization
To perform quantization, use the configuration file `phi3_nvmo_ptq.json`. This config executes two passes: one for model building and one for quantization. Note that ModelOpt currently only supports quantizing models created with the `modelbuild` tool.
```bash
olive run --config phi3_nvmo_ptq.json
```
## Steps to Quantize Different LLM Models
- **Locate and Update Configuration File:**
   Open `phi3_nvmo_ptq.json` in a text editor. Update the `model_path` to point to the directory or repository of the model you want to quantize. Ensure that `tokenizer_dir` is set to the tokenizer directory for the new model.

## More Inference Examples
- [Android chat APP with Phi-3 and ONNX Runtime Mobile](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile/examples/phi-3/android)

- [Web chat APP with Phi-3 and ONNX Runtime Web](https://github.com/microsoft/onnxruntime-inference-examples/tree/gs/chat/js/chat)
