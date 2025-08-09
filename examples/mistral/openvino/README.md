# Mistral Quantization

This folder contains a sample use case of Olive to optimize a Mistral-7B-Instruct models using OpenVINO tools.

- Intel® NPU: [Mistral 7B Instruct v0.2 Dynamic Shape Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Intel® GPU: [Mistral 7B Instruct v0.2 Dynamic Shape Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Intel® GPU: [Mistral 7B Instruct v0.3 Dynamic Shape Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Mistral Dynamic shape model

The following config files executes the above workflow producing as dynamic shaped model:

1. [Mistral-7B-Instruct-v0.2-npu-context-ov-dy-sym-gs128-bkp-int8-sym.json](Mistral-7B-Instruct-v0.2-npu-context-ov-dy-sym-gs128-bkp-int8-sym.json)
2. [Mistral_7B_Instruct_v0.2_gpu_context_ov_dy.json](Mistral_7B_Instruct_v0.2_gpu_context_ov_dy.json)
3. [Mistral-7B-Instruct-v0.3-gpu-context-ov-dy.json](Mistral-7B-Instruct-v0.3-gpu-context-ov-dy.json)


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
olive run --config Mistral-7B-Instruct-v0.3-gpu-context-ov-dy.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("<config_file.json>")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
