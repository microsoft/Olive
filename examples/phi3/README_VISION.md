# Phi3 optimization with Olive
This folder contains an example of optimizing the microsoft/Phi-3-vision-128k-instruct model from [Hugging Face](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) with Olive.

## Prerequisites
Install python packages
```
pip install -r requirements-vision.txt
```
Please install the following packages as you need:
- For CPU:
```bash
pip install onnxruntime>=1.19.2
```
- For CUDA 11.X:
```bash
pip install onnxruntime-gpu>=1.19.2 --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```
- For CUDA 12.X:
```bash
pip install onnxruntime-gpu>=1.19.2
```

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
