# MobileNet optimization with QDQ Quantization on Qualcomm NPU
This folder contains a sample use case of Olive to optimize a MobileNet model for Qualcomm NPU (QNN SDK).
https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk

**Note that**: This sample is supported on x86_64 Linux and Windows.

## Prerequisites
### Download and unzip QNN SDK
Download the QNN SDK and unzip the file.

Set the environment variable `QNN_SDK_ROOT` as `<qnn-sdk-unzipped-path>`

### Create an python environment for conversion/model library building
```
olive configure-qualcomm-sdk --py_version 3.8 --sdk qnn
```

### Prepare workflow config json
```
python prepare_config.py
```

### Pip requirements
Install the necessary python packages in your x64 python Olive environment:
```
python -m pip install -r requirements.txt
```

### Download data and model
To download the necessary data and model files using your x64 python Olive environment:
```
python download_files.py
```

## Run sample using config
In your x64 python Olive environment:

```
olive run --config raw_qnn_sdk_config.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("raw_qnn_sdk_config.json")
```
