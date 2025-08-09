# Phi 3 mini instruct Quantization

This folder contains a sample use case of Olive to optimize a Phi-3-mini-instruct models using OpenVINO tools.

- Intel® GPU: [Phi 3 Mini 4k Instruct Dynamic Shape Model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- Intel® NPU: [Phi 3 Mini 4k Instruct Dynamic Shape Model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- Intel® GPU: [Phi 3 mini 128k Instruct Dynamic Shape Model](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Phi 3 Dynamic shape model

The following config files executes the above workflow producing as dynamic shaped model:

1. [Phi-3-mini-4k-instruct-gpu-context-ov-dy.json](Phi-3-mini-4k-instruct-gpu-context-ov-dy.json)
2. [Phi-3-mini-4k-instruct-npu-context-ov-dynamic-sym-gs128-bkp-int8-sym.json](Phi-3-mini-4k-instruct-npu-context-ov-dynamic-sym-gs128-bkp-int8-sym.json)
3. [Phi-3-mini-128k-instruct-gpu-context-ov-dy.json](Phi-3-mini-128k-instruct-gpu-context-ov-dy.json)

## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model:

```bash
olive run --config <config_file.json>
```

Example:

```bash
olive run --config Phi-3-mini-4k-instruct-gpu-context-ov-dy.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("<config_file.json>")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
