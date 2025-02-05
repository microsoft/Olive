# MobileNet optimization with QDQ Quantization on Qualcomm NPU
This folder contains a sample use case of Olive to optimize a MobileNet model for Qualcomm NPU (QNN Execution Provider) using static QDQ quantization.

This example requires an x86 python environment on a Windows ARM machine.


## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Install onnxruntime-qnn
```bash
python -m pip install onnxruntime-qnn
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

## Run the sample
Run the following command to quantize the model and evaluate it on the NPU:
```bash
olive run --config mobilenet_qnn_ep.json
```

**NOTE:** The model optimization part of the workflow can also be done on a Linux/Windows machine with a different onnxruntime package installed. Remove the `"evaluators"` and `"evaluator"` sections from the `mobilenet_qnn_ep.json` configuration file to skip the evaluation step.
