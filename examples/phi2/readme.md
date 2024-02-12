# phi2 optimization with Olive
This folder contains an example of phi2 optimization with Olive workflow.

- CPU: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

## Prerequisites
* einops
* Pytorch>=2.2.0 and ORT nightly. Refer to https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers/models/phi2#prerequisites

## Usage\
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
