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
olive quantize -m microsoft/Phi-3-mini-4k-instruct --trust_remote_code --precision fp16 --implementation quarot

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

We also introduce QuaRot, a new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end.
Specific details about the algorithm can be found in the linked [paper](https://arxiv.org/pdf/2404.00456).

## Prerequisites
[QuaRot](https://github.com/microsoft/TransformerCompression/tree/quarot-main)

To run the workflow,
```bash
python phi3.py --quarot
```

### Get access to fine-tuning dataset
Get access to the following resources on Hugging Face Hub:
- [nampdn-ai/tiny-codes](https://huggingface.co/nampdn-ai/tiny-codes)

# Quantize using [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
The _TensorRT Model Optimizer_ is engineered to deliver advanced model compression techniques, including quantization, to Windows RTX PC systems. Specifically tailored to meet the needs of Windows users, it is optimized for rapid and efficient quantization, featuring local GPU calibration, reduced system and video memory consumption, and swift processing times.

The primary objective of the _TensorRT-Model-Optimizer_ is to generate optimized, standards-compliant ONNX-format models for DirectML backends

## Setup
```bash
pip install olive-ai[nvmo]
pip install onnxruntime-genai-directml>=0.4.0
pip install onnxruntime-directml==1.20.0
pip install -r requirements-nvmo-awq.txt
```
Install the CUDA version compatible with CuPy as mentioned in requirements-nvmo-awq.txt

## Validate Installation
After completing the setup, it's essential to verify that the `modelopt` package is installed correctly. Run the following command in your terminal:
```bash
python -c "from modelopt.onnx.quantization.int4 import quantize as quantize_int4"
```

## Quantization
For quantization, use the config file `phi3_nvmo_ptq.json`. This configuration runs **two passes**: one for the builder and one for quantization. Currently, Modelopt only supports quantizing models that were built with `modelbuild`; no other builder types are supported.

```bash
olive run --config phi3_nvmo_ptq.json
```

### Steps to Quantize a Different LLM Models

- **Locate and Update the Configuration File:**

    Open the `phi3_nvmo_ptq.json` file in your preferred text editor. Change the `model_path` to point to the repository or directory of the model you wish to quantize and ensure that the `tokenizer_dir` corresponds to the tokenizer used by the new model.

## More Inference Examples
- [Android chat APP with Phi-3 and ONNX Runtime Mobile](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile/examples/phi-3/android)

- [Web chat APP with Phi-3 and ONNX Runtime Web](https://github.com/microsoft/onnxruntime-inference-examples/tree/gs/chat/js/chat)
