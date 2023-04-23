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

## Follow the code sample to use olive output model
Please check code_sample.py for the sample how to use output model and inference_config.json for your inference. Find more details about ONNX Runtime Python API in https://onnxruntime.ai/docs/get-started/with-python.html.

The sample code works with ONNX Runtime 1.14.x and prior versions.
