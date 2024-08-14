# Phi3 optimization with Olive
This folder contains an example of optimizing the microsoft/Phi-3-vision-128k-instruct model from [Hugging Face](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) with Olive.

## Prerequisites
Please refer to the [pre-requisites of Phi3-vision optimization with GenAI](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi-3-vision.md#0-pre-requisites) for more details.
Basically, you need to install the following packages:
- `huggingface_hub[cli]`
- `numpy`
- `onnx`
- `ort-nightly>=1.19.0.dev20240601002` or `ort-nightly-gpu>=1.19.0.dev20240601002`
    - [ORT nightly package](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages) is needed until the latest changes are in the newest ORT stable package
    - For CPU:
    ```bash
    pip install ort-nightly --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
    ```
    - For CUDA 11.X:
    ```bash
    pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
    ```
    - For CUDA 12.X:
    ```bash
    pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
    ```
- `pillow`
- `requests`
- `torch`
- `torchvision`
- `transformers`


### For optimizing model from Hugging Face
if you have not logged in Hugging Face account,
- Install Hugging Face CLI and login your Hugging Face account for model access
```
huggingface-cli login
```

## Usage
we will use the `phi3_vision.py` script to optimize model for a chosen hardware target by running the following commands.

```
python phi3.py [--target HARDWARE_TARGET(cpu|cuda)] [--precision DATA_TYPE] [--output_dir OUTPUT_DIR] [--cache_dir CACHE_DIR] [--inference] [--optimized_model_path OPT_MODEL_PATH]

# Examples
# optimize model and start a inference process
python phi3_vision.py --cache_dir cache --output_dir output --inference

# inference with optimized model
python phi3_vision.py --optimized_model_path optimized_model_path --inference
```
