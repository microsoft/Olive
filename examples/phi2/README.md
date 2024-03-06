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
In this stage, we will use the `phi2.py` script to generate optimized models and do inference with the optimized models.

Following are the model types that can be used for optimization:
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

Above commands will generate optimized models with given model_type and save them in the `phi2` folder. These optimized models can be wrapped by ONNXRuntime for inference.
Besides, for better generation experience, this example also let use use [Optimum](https://huggingface.co/docs/optimum/v1.2.1/en/onnxruntime/modeling_ort) to generate optimized models.
Then use can call `model.generate` easily to run inference with the optimized model.
```bash
# optimum optimization
python phi2.py --model_type cpu_fp32 --optimum_optimization
```

Then let us use the optimized model to do inference.

## Generation example of optimized model
```bash
# --prompt is optional, can accept a string or a list of strings
# if not given, the default prompt "Write a function to print 1 to n" "Write a extremely long story starting with once upon a time"
python phi2.py --model_type cpu_fp32 --inference --prompt "Write a extremely long story starting with once upon a time"
```
This command will
1. generate optimized models if you never run the command before,
2. reuse the optimized models if you have run the command before,
3. then use the optimized model to do inference with greedy Top1 search strategy.
Note that, we only use the simplest greedy Top1 search strategy for inference example which may show not very reasonable results.

For better generation experience, here is the way to run inference with the optimized model using Optimum.
```bash
python phi2.py --model_type cpu_fp32 --inference --optimum_optimization --prompt "Write a extremely long story starting with once upon a time"
```


## Limitations
1. The latest ONNXRuntime implements specific fusion patterns for better performance but only works for ONNX model from TorchDynamo-based ONNX Exporter. And the TorchDynamo-based ONNX Exporter is only available on Linux.
When using Windows, this example will fallback to the default PyTorch ONNX Exporter, that can achieve a few improvements but not as much as the TorchDynamo-based ONNX Exporter.
Therefore, it is recommended to use Linux for phi2 optimization.

2. For Optimum optimization, the dynamo model is not supported very well. So we use legacy Pytorch ONNX Exporter to run optimization like what we do in Windows.
