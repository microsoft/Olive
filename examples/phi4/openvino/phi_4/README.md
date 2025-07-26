# Phi-4 Quantization

This folder contains a sample use case of Olive to optimize a [microsoft/Phi-4](https://huggingface.co/microsoft/Phi-4) model using OpenVINO tools.

- Intel® GPU: [Phi4 Dynamic shape model](#phi_4_gpu_context_dy)

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Dynamic shape model

The workflow in Config file: [phi_4_gpu_context_dy.json](phi_4_gpu_context_dy.json) executes the above workflow producing a dynamic shape model.

## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model

```bash
olive run --config phi_4_gpu_context_dy.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("phi_4_gpu_context_dy.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

The sample command to run after all setup would be as follows:-

```bash
python model-chat.py -e follow_config -v -g -m models/Phi_4_gpu_context_dy/model/
```
