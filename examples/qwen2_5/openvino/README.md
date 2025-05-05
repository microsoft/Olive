# Qwen-2.5-1.5B-Instruct Quantization

This folder contains a sample use case of Olive to optimize a [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model using OpenVINO tools.

- Intel® NPU: [Qwen2.5 1.5B Dynamic shape model sym bkp int8 sym r1](#dynamic-shape-model-sym-bpk-int8-sym-r1)

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Dynamic shape model sym bkp int8 sym r1

The workflow in Config file: [Qwen2.5-1.5B-instruct_context_ov_dynamic_sym_bkp_int8_sym_r1.json](Qwen2.5-1.5B-instruct_context_ov_dynamic_sym_bkp_int8_sym_r1.json) executes the above workflow producing a dynamic shape model.

## How to run

### Pip requirements

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model

```bash
olive run --config Qwen2.5-1.5B-instruct_context_ov_dynamic_sym_bkp_int8_sym_r1.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("Qwen2.5-1.5B-instruct_context_ov_dynamic_sym_bkp_int8_sym_r1.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
