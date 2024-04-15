# Bert model optimization on Qualcomm NPU with SNPE SDK
This folder contains a sample use case of Olive to convert an bert model Onnx model, then to SNPE DLC and to evaluate the accuracy of the DLC model.

Performs optimization pipeline:
- *Pytorch Model -> Onnx Model with Dynamic Shape -> Onnx Model with Fixed Shape -> SNPE Model*

## Prerequisites
### Download and unzip SNPE SDK
Download the SNPE SDK zip following [instructions from Qualcomm](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)

We test it with SNPE v2.18.0.240101.

Unzip the file and set the unzipped directory path as environment variable `SNPE_ROOT`.

### Configure SNPE
```sh
olive configure-qualcomm-sdk --py_version 3.8 --sdk snpe
```

## Run sample
Run the conversion and quantization locally.
```
olive run --config bert_snpe.json
```

## Issues

1. "Module 'qti.aisw.converters' has no attribute 'onnx':
    Refer to this: https://developer.qualcomm.com/comment/21810#comment-21810,
    change the import statement in `{SNPE_ROOT}/lib/python/qti/aisw/converters/onnx/onnx_to_ir.py:L30` to:
    ```python
    from qti.aisw.converters.onnx import composable_custom_op_utils as ComposableCustomOp
    ```
