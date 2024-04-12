# MobileNet optimization with QDQ Quantization on Qualcomm NPU
This folder contains a sample use case of Olive to optimize a MobileNet model for Qualcomm NPU (QNN Execution Provider)
using static QDQ quantization.

## Prerequisites for Quantization
### Clone the repository and install Olive (x86 python)

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Install onnxruntime (x86 python)
This example requires onnxruntime>=1.17.0. Please install the latest version of onnxruntime:

```bash
python -m pip install "onnxruntime>=1.17.0"
```

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Prerequisites for Evaluation

### Download and unzip QNN SDK
Download the Qualcomm AI Engine Direct SDK (QNN SDK) from https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct.

Complete the steps to configure the QNN SDK for QNN EP as described in the [QNN EP Documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#running-a-quantized-model-on-windows-arm64).

Set the environment variable `QNN_LIB_PATH` as `QNN_SDK\lib\aarch64-windows-msvc`.

### Install onnxruntime-qnn (ARM64 python)
If you want to evaluate the quantized model on the NPU, you will need to install the onnxruntime-qnn package. This package is only available for Windows ARM64 python so you will need a separate ARM64 python installation to install it.

Using an ARM64 python installation, create a virtual environment and install the onnxruntime-qnn package:
```bash
python -m venv qnn-ep-env
qnn-ep-env\Scripts\activate
python -m pip install ort-nightly-qnn --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
deactivate
```

Set the environment variable `QNN_ENV_PATH` to the directory where the python executable is located:
```bash
set QNN_ENV_PATH=C:\path\to\qnn-ep-env\Scripts
```

**Note:** Using a virtual environment is optional but recommended to better manage the dependencies.

## Run the sample

### Quantize the model
Run the following command to quantize the model:
```bash
python mobilenet_qnn_ep.py
```

### Quantize and evaluate the model
Run the following command to quantize the model and evaluate it on the NPU:
```bash
python mobilenet_qnn_ep.py --evaluate
```

**NOTE:** You can also only dump the workflow configuration file by adding the `--config_only` flag to the command.

The configuration file will be saved as `mobilenet_qnn_ep_{eval|no_eval}.json` in the current directory and can be run using the Olive CLI.
```bash
olive run --config mobilenet_qnn_ep_{eval|no_eval}.json
```
