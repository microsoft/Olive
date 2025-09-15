# Llama2 Optimization using AutoFusion

This folder contains sample use cases of Olive to optimize a [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf) model using AutoFusion.

AutoFusion automatically finds compatible chains of operators in an onnx model and fuses them into a single operator. This
operator is implemented as a custom op in ONNX Runtime using an ahead-of-time compiled triton kernel. The triton kernel is
created using a codegen tool.

Performs optimization pipeline:
- GPU, FP32: *PyTorch Model -> Onnx Model -> Onnx Model with AutoFusion*

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive. Only tested with Python 3.8.

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
pip install triton-nightly==2.1.0.post20240108192258 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/
```

## Set CUDA_HOME
Set the path to cuda as `CUDA_HOME` environment variable. Only tested with CUDA 12.2 and 12.5.

## Run the example
```bash
python -m olive.workflows.run --config llama2_auto_fusion.json
```

## Notes
- The triton compilation required `libxcrypt`. If it doesn't exist, install it using:
```bash
# ubuntu
sudo apt-get install libxcrypt-dev

# Azure Linux
sudo dnf install libxcrypt-devel
```