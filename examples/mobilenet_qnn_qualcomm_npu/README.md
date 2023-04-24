# MobileNet optimization with QDQ Quantization on Qualcomm NPU
This folder contains a sample use case of Olive to optimize a MobileNet model for Qualcomm NPU (QNN Execution Provider)
using static QDQ quantization tuner.

## Prerequisites
### Download and unzip QNN SDK
Download the QNN SDK and unzip the file.

Set the environment variable `QNN_LIB_PATH` as `<qnn-sdk-unzipped-path>\target\aarch64-windows-msvc\lib`

### Create an arm64 python environment with onnxruntime-qnn
Olive is not supported on arm64 version of python so we require the onnxruntime-qnn package to be installed in a separate arm64 environment. This environment can be a [python virtual env](https://docs.python.org/3/library/venv.html).

Set the path to the directory with arm64 python executable as environment variable `QNN_ENV_PATH`

For example if you used a python venv located at `C:\qnn\qnn-venv`, then `QNN_ENV_PATH` is `C:\qnn\qnn-venv\Scripts`

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
python -m olive.workflows.run --config mobilenet_config.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("mobilenet_config.json")
```
