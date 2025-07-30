# Phi-4-mini-reasoning Quantization

This folder contains a sample use case of Olive to optimize a [microsoft/Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning) model using OpenVINO tools.

- Intel® NPU: [Phi4 Mini Reasoning Dynamic shape model sym_gs128 bkp int sym](#dynamic-shape-model-sym-gs128-bkp-int8-sym)
- Intel® GPU: [Phi4 Mini Reasoning GPU Dynamic shape model ](#Phi_4_mini_reasoning_gpu_context_dy)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Phi 4 mini reasoning dynamic shape model

The following config files execute the above workflow, producing a dynamic shaped model:

1. [Phi-4-mini-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym.json](Phi-4-mini-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym.json)
2. [Phi_4_mini_reasoning_gpu_context_dy.json](Phi_4_mini_reasoning_gpu_context_dy.json)

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
olive run --config Phi-4-mini-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("Phi-4-mini-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

The sample command to run after all setup would be as follows:-

```bash
python model-chat.py -e follow_config -v -g -m models/Phi-4-mini-reasoning_context_ov_dynamic_sym_gs128_bkp_int8_sym/model/
```
