# DeepSeek-R1-Distill-Qwen Quantization

This folder contains a sample use case of Olive to optimize a DeepSeek-R1-Distill-Qwen models using OpenVINO tools.

- Intel® NPU: [DeepSeek R1 Distill Qwen 1.5B Dynamic Shape Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- Intel® GPU: [DeepSeek R1 Distill Qwen 1.5B Dynamic Shape Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- Intel® GPU: [DeepSeek R1 Distill Qwen 7B Dynamic Shape Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- Intel® GPU: [DeepSeek R1 Distill Qwen 14B Dynamic Shape Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- Intel® GPU: [DeepSeek R1 Distill Llama 8B Dynamic Shape Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)


## Quantization Workflows

This workflow performs quantization with Optimum Intel®. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Deepseek R1 Distill Qwen Dynamic shape model

The following config files execute the above workflow, producing a dynamic shape model:

1. [DeepSeek-R1-Distill-Qwen-1.5B-npu-context-ov-dy-sym-gs128-bkp-int8-sym-r1.json](DeepSeek-R1-Distill-Qwen-1.5B-npu-context-ov-dy-sym-gs128-bkp-int8-sym-r1.json)

2. [DeepSeek-R1-Distill-Qwen-1.5B-gpu-context-ov-dy-gs32-r1.json](DeepSeek-R1-Distill-Qwen-1.5B-gpu-context-ov-dy-gs32-r1.json)

3. [DeepSeek-R1-Distill-Qwen-7B-gpu-context-ov-dy-gs128-r1.json](DeepSeek-R1-Distill-Qwen-7B-gpu-context-ov-dy-gs128-r1.json)

4. [DeepSeek-R1-Distill-Qwen-14B-gpu-context-ov-dy-gs128-r1.json](DeepSeek-R1-Distill-Qwen-14B-gpu-context-ov-dy-gs128-r1.json)

5. [DeepSeek-R1-Distill-Llama-8B-gpu-context-ov-dy-gs128-r1.json](DeepSeek-R1-Distill-Llama-8B-gpu-context-ov-dy-gs128-r1.json)

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
olive run --config <config_file.json>
```

Example:

```bash
olive run --config DeepSeek-R1-Distill-Qwen-1.5B-npu-context-ov-dy-sym-gs128-bkp-int8-sym-r1.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("<config_file.json>")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.

### (Optional) Run Console-Based Chat Interface

To run ONNX OpenVINO IR Encapsulated GenAI models, please setup latest ONNXRuntime GenAI with ONNXRuntime OpenVINO EP support.

The sample chat app to run is found as [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py) in the [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai/) Github repository.

The sample command to run after all setup would be as follows:-

```bash
python model-chat.py -e follow_config -v -g -m models/<model_folder>/model/
```

Example:

```bash
python model-chat.py -e follow_config -v -g -m models/DeepSeek-R1-Distill-Qwen-1.5B_context_ov_dynamic_sym_gs128_bkp_int8_sym_r1/model/
```
