# Olive output instruction

## ONNXRuntime installation
### Option 1: follow the installation instruction
Please follow the instruction to install ONNX Runtime: https://onnxruntime.ai/docs/install/#cccwinml-installs

### Option 2: install by local NuGet file
You can find ONNXRuntime NuGet package file in `ONNXRuntime` folder. Install the local NuGet file by:
```
Install-Package <absolute-path-to-your-output-folder>\SampleCode\cs\ONNXRuntime\<onnxruntime-package-name>.nupkg
```

## Follow the code sample to use olive output model
Please check code_sample.cpp for the sample how to use output model and inference_config.json for your inference. Find more details about ONNX Runtime C# API in https://onnxruntime.ai/docs/get-started/with-csharp.html.

The sample code works with ONNX Runtime 1.14.x and prior versions.
