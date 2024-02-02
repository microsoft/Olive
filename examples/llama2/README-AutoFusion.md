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
This example requires the latest nightly build of ort-nightly-gpu
```bash
# Uninstall previous onnxruntime packages
pip uninstall -y onnxruntime-gpu onnxruntime ort-nightly-gpu ort-nightly
# Install latest nightly build of ort-nightly-gpu
pip install ort-nightly-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
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

## Clone ONNX Runtime repository
```bash
git clone https://github.com/microsoft/onnxruntime.git onnxruntime-repo
# set ONNXRUNTIME_DIR to the path of the cloned repository
export ONNXRUNTIME_DIR=$PWD/onnxruntime-repo
```

Also requires the path to cuda to be set using `CUDA_HOME` environment variable. Only tested with CUDA 12.2

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
