# VGG model optimization on Qualcomm NPU
This folder contains a sample use case of Olive to convert an Onnx model to SNPE DLC, quantize it and convert it to Onnx.

Performs optimization pipeline:
- *Onnx Model -> SNPE Model -> Quantized SNPE Model*

## Prerequisites
### Download and unzip SNPE SDK
Download the SNPE SDK zip following [instructions from Qualcomm](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)

We test it with SNPE v2.18.0.240101.

Unzip the file and set the unzipped directory path as environment variable `SNPE_ROOT`.

### Configure SNPE
```sh
# in general, python 3.9+ is recommended
olive configure-qualcomm-sdk --py_version 3.9 --sdk snpe
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

## Prepare the configuration file
The configuration file `vgg_config.json` contains the parameters for the conversion and quantization.
But the quantization is only supported for `x64-Linux` platform. To run the optimization on Windows, please remove the `quantization` section from the configuration file.

Or you can just run following command to generate the configuration file:
```sh
python prepare_config.py
```

## Run sample
Run the conversion and quantization locally. Only supports `x64-Linux`.
```
olive run --config vgg_config.json
```

## Issues

1. "Module 'qti.aisw.converters' has no attribute 'onnx':
    Refer to this: https://developer.qualcomm.com/comment/21810#comment-21810,
    change the import statement in `{SNPE_ROOT}/lib/python/qti/aisw/converters/onnx/onnx_to_ir.py:L30` to:
    ```python
    from qti.aisw.converters.onnx import composable_custom_op_utils as ComposableCustomOp
    ```
