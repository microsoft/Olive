# Llama2 Optimization using AutoFusion

This folder contains sample use cases of Olive to optimize a [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf) model using AutoFusion.

AutoFusion automatically finds compatible chains of operators in an onnx model and fuses them into a single operator. This
operator is implemented as a custom op in ONNX Runtime using an ahead-of-time compiled triton kernel. The triton kernel is
created using a codegen tool.

Performs optimization pipeline:
- GPU, FP32: *PyTorch Model -> Onnx Model -> Onnx Model with AutoFusion*

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

## Install onnxruntime
This example requires onnxruntime-gpu>=1.17.0. Please install the latest version of onnxruntime:

```bash
python -m pip install "onnxruntime-gpu>=1.17.0"
```

### Install extra dependencies
Install the necessary python packages:
```bash
python -m pip install -r requirements.txt
```

## Install triton-nightly
This example requires the latest nightly build of triton. Note: This needs to be done last to avoid reinstalling the stable version of triton.
```bash
# Uninstall previous triton packages
pip uninstall -y triton triton-nightly
# Install latest nightly build of triton-nightly
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

## Set CUDA_HOME
Set the path to cuda as `CUDA_HOME` environment variable. Only tested with CUDA 12.2

## Run the example
```bash
python -m olive.workflows.run --config llama2_auto_fusion.json
```

**Note**: If you want to package the model, custom operator library and sample code into a zip file, add the following config option under `"engine"` in the config file:
```json
"packaging_config": {
    "type": "Zipfile",
    "name": "open_llama_3b_auto_fusion"
}
```
It might take a while to compress the files into a zip file.
