# Olive output instruction

## ONNXRuntime installation
### Option 1: install by pip
Install onnxruntime package:
```
python -m pip install onnxruntime
```

### Option 2: install by local wheel
You can find ONNXRuntime wheel file is in `ONNXRuntime` folder. Install the local wheel file by pip:
```
python -m pip install ONNXRuntime/<onnxruntime_package_name.whl>
```

The local package can be either an `onnxruntime` package, or an `onnxruntime-gpu` package, or both. This depends on the `execution_provider` in `inference_config.json` file.

If `execution_provider` includes `CUDAExecutionProvider`, then the `onnxruntime-gpu` package should be used for your inference, and the `onnxruntime-gpu` package will be included in the output folder.

If `execution_provider` includes `CPUExecutionProvider` or other EPs, then the `onnxruntime` package will be included in the output folder.

Please follow the ONNXRuntime Execution Provider instruction [link](https://onnxruntime.ai/docs/execution-providers/) to setup and configure your Execution Provider based on the `execution_provider` in the `inference_config.json` file.

## Follow the code sample to use olive output model
Please check code_sample.py for the sample how to use output model and inference_config.json for your inference. Find more details about ONNX Runtime Python API in https://onnxruntime.ai/docs/get-started/with-python.html.

The sample code works with ONNX Runtime 1.14.x and prior versions.


## ONNXRuntime Extensions
Onnxruntime extensions package could be included if the ONNX model is using onnxruntime-extensions.

### Option 1: install by pip
Install onnxruntime-extensions package:
```
python -m pip install onnxruntime-extensions
```

### Option 2: install by local wheel
* For Windows
  Please install it by "`python -m pip install ONNXRuntime/<onnxruntime-extensions.whl>`"
* For Linux
    * Nightly
      The packages are not ready yet, so it could be installed from source. Please make sure the compiler toolkit like gcc(later than g++ 8.0) or clang, and the tool cmake are installed before the following command:
      `python -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git`
    * Stable
      Please install it by `python -m pip install ONNXRuntime/<onnxruntime-extensions.whl>`
