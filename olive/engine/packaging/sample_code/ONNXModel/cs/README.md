# Olive output instruction

## ONNXRuntime installation
### Option 1: follow the installation instruction
Please follow the instruction to install ONNX Runtime: https://onnxruntime.ai/docs/install/#cccwinml-installs

### Option 2: install by local NuGet file
You can find ONNXRuntime NuGet package file in `ONNXRuntime` folder. Install the local NuGet file by:
```
dotnet add package <absolute-path-to-your-output-folder>\SampleCode\cpp\ONNXRuntime\<onnxruntime-package-name>.nupkg -s local
```

The local package can be either an `onnxruntime` package, or an `onnxruntime-gpu` package, or both. This depends on the `execution_provider` in `inference_config.json` file.

If `execution_provider` includes `CUDAExecutionProvider`, then the `onnxruntime-gpu` package should be used for your inference, and the `onnxruntime-gpu` package will be included in the output folder.

If `execution_provider` includes `CPUExecutionProvider` or other EPs, then the `onnxruntime` package will be included in the output folder.

Please follow the ONNXRuntime Execution Provider instruction [link](https://onnxruntime.ai/docs/execution-providers/) to setup and configure your Execution Provider based on the `execution_provider` in the `inference_config.json` file.

## Follow the code sample to use olive output model
Please check code_sample.cpp for the sample how to use output model and inference_config.json for your inference. Find more details about ONNX Runtime C# API in https://onnxruntime.ai/docs/get-started/with-csharp.html.

The sample code works with ONNX Runtime 1.14.x and prior versions.
