# phi2 optimization with Olive
This folder contains an example of phi2 optimization with Olive workflow.

- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

## Prerequisites
* einops
* Pytorch>=2.2.0 \
  _The [official website](https://pytorch.org/) offers packages compatible with CUDA 11.8 and 12.1. Please select the appropriate version according to your needs._
* [ONNXRuntime nightly package](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages)

## Usage
cpu_fp32
```bash
python phi2.py --cpu_fp32
```
cpu_int4
```bash
python phi2.py --cpu_int4
```
cuda_fp16
```bash
python phi2.py --cuda_fp16
```
cuda_int4
```bash
python phi2.py --cuda_int4
```

## Limitations
TorchDynamo-based ONNX Exporter only supports Linux.
