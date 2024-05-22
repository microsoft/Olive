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


### For optimizing model from Hugging Face
if you have not loged in Hugging Face account,
- Install Hugging Face CLI and login your Hugging Face account for model access
```
huggingface-cli login
```

### For optimizing model from Azure Machine Learning Model Catalog

- Install Olive with Azure Machine Learining dependency
```
pip install olive-ai[azureml]
```
if you have not loged in Azure account,
- Install Azure Command-Line Interface (CLI) following [this link](https://learn.microsoft.com/en-us/cli/azure/)
- Run `az login` to login your Azure account to allows Olive to access the model.


## Usage
we will use the `phi3.py` script to generate optimized model for a chosen hardware target by running the following commands.

```
python phi3.py [--target HARDWARE_TARGET] [--precision DATA_TYPE] [--source SOURCE] [--inference] [--prompt PROMPT] [--max_length LENGTH]

# Examples
python phi3.py --target mobile

python phi3.py --target model --source AzureML

python phi3.py --target mobile --inference --prompt "Write a story starting with once upon a time" --max_length 200
```

- `--target`: cpu, cuda, mobile, web
- `--precision`: optional, for data precision. fp32 or int4 (default) for cpu target; fp32, fp16, or int4 (default) for GPU target; int4 (default) for mobile or web.
- `--source`: optional, for model path. HF or AzureML. HF(Hugging Face model) by default.
- `--inference`: optional, for non-web models inference/validation.
- `--prompt`: optional, the prompt text fed into the model. Take effect only when `--inference` is set.
- `--max_length`: optional, the max length of the output from the model. Take effect only when `--inference` is set.


This script includes
- Generate the Olive configuration file for the chosen HW target
- Generate optimized model with Olive based on the configuration file for the chosen HW target
- (optional) Inference the optimized model with ONNX Runtime Generate() API with non-web target


If you have an Olive configuration file, you can also run the olive command for model generation:
```
olive run [--config CONFIGURATION_FILE]

# Examples
olive run --config phi3_mobile_int4.json
```
