# Phi-3.5-mini-instruct Quantization

This folder contains a sample use case of Olive to optimize a Phi-3.5-mini-instruct model using OpenVINO tools.

- Intel® NPU: [Phi 3.5 Mini Instruct Dynamic Shape Model](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- Intel® GPU: [Phi 3.5 Mini Instruct Dynamic Shape Model](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Phi 3.5 Mini Instruct Dynamic Shape Model

The flow in the following config files executes the above workflow producing a dynamic shape model.

1. [Phi-3.5-mini-instruct-npu-context-ov-dynamic-sym-gs128-bkp-int8-sym.json](Phi-3.5-mini-instruct-npu-context-ov-dynamic-sym-gs128-bkp-int8-sym.json)
2. [Phi-3.5-mini-instruct-gpu-context-ov-dynamic.json](Phi-3.5-mini-instruct-gpu-context-ov-dynamic.json)

## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model using the following command:

```bash
olive run --config <config_file.json>
```

Example:

```bash
olive run --config Phi-3.5-mini-instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("<config_file.json>")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

The sample command to run after all setup would be as follows:

```bash
python model-chat.py -e follow_config -v -g -m models/<model_folder>/model/
```

Example:

```bash
python model-chat.py -e follow_config -v -g -m models/Phi-3.5-mini-instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym/model/
```
