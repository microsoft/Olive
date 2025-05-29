# Llama-3.2-1B-Instruct Quantization

This folder contains a sample use case of Olive to optimize a [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model using OpenVINO tools.

## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Dynamic shape model

The workflow in Config file: [Llama-3.2-1B-Instruct_context_ov_dynamic_sym_bkp_int8_sym.json](Llama-3.2-1B-Instruct_context_ov_dynamic_sym_bkp_int8_sym.json) executes the above workflow producing a dynamic shape model.

## How to run

### Setup

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

**NOTE:**

- Access to the [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model is gated and therefore you will need to request access to the model. Once you have access to the model, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model

```bash
olive run --config Llama-3.2-1B-Instruct_context_ov_dynamic_sym_bkp_int8_sym.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("Llama-3.2-1B-Instruct_context_ov_dynamic_sym_bkp_int8_sym.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

The sample command to run after all setup would be as follows:-

```bash
python model-chat.py -e follow_config -v -g -m models/Llama-3.2-1B-Instruct_context_ov_dynamic_sym_bkp_int8_sym/model/
```
