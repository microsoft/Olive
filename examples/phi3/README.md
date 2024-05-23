# Phi3 optimization with Olive
This folder contains an example of fine-tuning and optimizing [the Phi-3-Mini-4K-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model in HF for different hardware targets with Olive.


## Prerequisites
* einops
* Pytorch: >=2.2.0 \
  _The [official website](https://pytorch.org/) offers packages compatible with CUDA 11.8 and 12.1. Please select the appropriate version according to your needs._
* [Package onnxruntime](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages): >=1.18.0
* [Package onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai): >=0.2.0. If you target GPU, pls install onnxruntime and onnxruntime-genai gpu packages.

Install the dependencies
```
pip install -r requirements.txt
```

## Usage
we will use the `phi3.py` script to fine-tune and optimize model for a chosen hardware target by running the following commands.

```sh
python phi3.py [--target HARDWARE_TARGET] [--finetune_method METHOD] [--precision DATA_TYPE] [--inference] [--prompt PROMPT] [--max_length LENGTH]

# Examples
python phi3.py --target web

python phi3.py --target mobile --inference --prompt "Write a story starting with once upon a time" --max_length 200

python phi3.py --target cuda --finetune_method lora --inference --prompt "Write a story starting with once upon a time" --max_length 200
# qlora introduce the quantization into base model which is not supported by onnxruntime-genai as of now!
python phi3.py --target cuda --finetune_method qlora
```

- `--target`: cpu, cuda, mobile, web
- `--finetune_method`: optional. The method used for fine-tuning. Options: `qlora`, `lora`. Default is none. Note that onnxruntime-genai only supports `lora` method as of now.
- `--precision`: optional. fp32, fp16, int4. fp32 or int4(default) for cpu target; fp32 or fp16 or int4(default) for gpu target; int4(default) for mobile or web
- `--inference`: run the optimized model, for non-web models inference.
- `--prompt`: optional, the prompt text fed into the model. Take effect only when `--inference` is set.
- `--max_length`: optional, the max length of the output from the model. Take effect only when `--inference` is set.


This script includes
1. Generate the Olive configuration file for your need including the chosen HW target, the preferred model precision.
2. Fine-tune model by lora or qlora method with dataset of `nampdn-ai/tiny-codes`.
3. Generate optimized model with Olive based on the configuration file for the chosen HW target
4. (optional) Inference the optimized model with ONNX Runtime Generation API. Not supported for web target


If you have an Olive configuration file, you can also run the olive command for model generation:
```
olive run [--config CONFIGURATION_FILE]

# Examples
olive run --config phi3_mobile_int4.json
```
