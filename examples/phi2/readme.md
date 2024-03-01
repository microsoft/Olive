# phi2 optimization with Olive
This folder contains an example of phi2 optimization with Olive workflow.

- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

## Prerequisites
* einops
* Pytorch>=2.2.0 \
  _The [official website](https://pytorch.org/) offers packages compatible with CUDA 11.8 and 12.1. Please select the appropriate version according to your needs._
* [ONNXRuntime nightly package](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages)
  In Linux, phi2 optimization requires the ONNXRuntime nightly package(>=1.18.0). In Windows, ONNXRuntime>=1.17.0 is recommended.

## Optimization Usage
cpu_fp32
```bash
python phi2.py --model_type cpu_fp32
```
cpu_int4
```bash
python phi2.py --model_type cpu_int4
```
cuda_fp16
```bash
python phi2.py --model_type cuda_fp16
```
cuda_int4
```bash
python phi2.py --model_type cuda_int4
```
Above commands will generate optimized models with given model_type and save them in the `phi2` folder. Then let us use the optimized model to do inference.

## Generation example of optimized model
```bash
# --prompt is optional, can accept a string or a list of strings
# if not given, the default prompt "Write a function to print 1 to n" "Write a extremely long story starting with once upon a time"
python phi2.py --model_type cpu_fp32 --inference --prompt "Write a extremely long story starting with once upon a time"
```
This command will
1. generate optimized models if you never run the command before,
2. reuse the optimized models if you have run the command before,
3. then use the optimized model to do inference with simple greedy Top1 search strategy.
Note that, we only use the simplest greedy Top1 search strategy for inference example which may show not very reasonable results.

## Limitations
The latest ONNXRuntime implements specific fusion patterns for better performance but only works for ONNX model from TorchDynamo-based ONNX Exporter. And the TorchDynamo-based ONNX Exporter is only available on Linux.
When using Windows, this example will fallback to the default PyTorch ONNX Exporter, that can achieve a few improvements but not as much as the TorchDynamo-based ONNX Exporter.
Therefore, it is recommended to use Linux for phi2 optimization.
