# MobileNet optimization with QDQ Quantization on Qualcomm NPU
This folder contains a sample use case of Olive to optimize a MobileNet model for Qualcomm NPU (QNN Execution Provider)
using static QDQ quantization.

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Install onnxruntime
This example requires onnxruntime>=1.17.0. Please install the latest version of onnxruntime:

For CPU:
```bash
python -m pip install "onnxruntime>=1.17.0"
```

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Download data and model
To download the necessary data and model files:
```
python download_files.py
```

## Run sample using config
```
python -m olive.workflows.run --config mobilenet_qnn_ptq.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("mobilenet_qnn_ptq.json")
```

**Note:**
- This sample currently only quantizes the model for QNN Execution Provider. It does not include evaluation of the quantized model on the NPU. Evaluation support will be added in the future.
- We use onnxruntime cpu package during the quantization process since it does not require inference using QNN EP. onnxruntime-qnn is only available for Windows ARM64 python but the dev dependencies are not all available for ARM64 python yet.
