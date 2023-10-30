# Llama2 optimization using ORT toolchain
This folder contains a sample use case of Olive to optimize a [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf) model using ONNXRuntime tools.

Performs optimization pipeline:
- CPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32*
- CPU, INT8: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32 -> Onnx Dynamic Quantization*
- CPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32 -> Onnx Block wise int4 Quantization*
- GPU, FP32: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp32*
- GPU, FP16: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 + Grouped Query Attention*
- GPU, INT4: *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16 + Grouped Query Attention -> Onnx Block wise int4 Quantization*

**Note that**: Currently, grouped query attention is only supported on GPU with fp16 and it requires the cuda architecture >= 80. You can just set `use_gqa` to `false` in the config file to disable it.
```json
"transformers_optimization_fp16": {
    "type": "OrtTransformersOptimization",
    "disable_search": true,
    "evaluator": "gqa_evaluator",
    "config": {
        "save_as_external_data": true,
        "all_tensors_to_one_file": true,
        "model_type": "gpt2",
        "opt_level": 0,
        "only_onnxruntime": false,
        "keep_io_types": false,
        "float16": true,
        "use_gqa": false // <----------- disable gqa
    }
}
```

## Prerequisites
### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.

### Install onnxruntime
Also we need latest version of onnxruntime which provides the support of int4 quantization/grouped query attention. Please install the latest version of onnxruntime:

1. From source:
    ```bash
    git clone https://github.com/microsoft/onnxruntime
    # compile ort with cuda support, which requires the image with cuda and cudnn installed
    bash ./build.sh \
        --config=Release \
        --build_dir="./test_build" \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --cuda_version=11.7 \
        --use_cuda --update --build \
        --build_wheel \
        --parallel \
        --skip_tests --cmake_extra_defines ONNXRUNTIME_VERSION=(cat ./VERSION_NUMBER) \CMAKE_CUDA_ARCHITECTURES="70;75;80" \
        --use_mpi=false
    ```
Then you can find the wheel file under folder of `build_dir`(`test_build/Release/dist/` in this case).

2. From nightly-build:

    Installation package table: https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages

After installation, you can run the following command to check if the onnxruntime is installed successfully:
```python
import onnxruntime as ort
ort.get_available_providers()  # should contain 'CUDAExecutionProvider'
```

### Install extra dependencies
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run the config to optimize the model
You can only generate the optimized config file by running the following command for double checking before running the optimization pipeline:
```bash
python llama2.py --model_name meta-llama/Llama-2-7b-hf --only_flag
```

Or you can run the following command to directly optimize the model:

CPU:
```bash
# run to optimize the model: FP32/INT8/INT4
python llama2.py --model_name meta-llama/Llama-2-7b-hf
```

GPU:
```bash
# run to optimize the model: FP32/INT8/INT4
python llama2.py --model_name meta-llama/Llama-2-7b-hf --gpu
# use gqa instead of mha
python llama2.py --model_name meta-llama/Llama-2-7b-hf --gpu --use_gqa
```

## TODO
- [ ] Add generation example of the optimized model.
- [ ] Attach the benchmark results.
