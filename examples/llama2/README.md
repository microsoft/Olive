# Whisper optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf) model using ONNXRuntime tools.

Performs optimization pipeline:
- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32 -> Onnx Dynamic Quantization*
- CPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32 -> Onnx Block wise int4 Quantization*
- GPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32*
- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 + Grouped Query Attention*
- GPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 + Grouped Query Attention -> Onnx Block wise int4 Quantization*

Outputs the final model and latency results.

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run the config to optimize the model
First, install required packages according to passes.

CPU:
```bash
# setup related packages
python -m olive.workflows.run --config ort_converter_merged_llama2_cpu.json --setup

# run to optimize the model: FP32/INT8/INT4
python -m olive.workflows.run --config ort_converter_merged_llama2_cpu.json
```

GPU:
```bash
# setup related packages
python -m olive.workflows.run --config ort_converter_merged_llama2_gpu.json --setup

# run to optimize the model: FP32/INT8/INT4
python -m olive.workflows.run --config ort_converter_merged_llama2_gpu.json
```


## TODO
- [ ] Add generation example of the optimized model.
- [ ] Attach the benchmark results.
