# Phi-3.5-mini-instruct Quantization

This folder contains a sample use case of Olive to optimize a [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model using OpenVINO tools.

- Intel® NPU: [Phi3.5 dynamic shape model sym_gs128 bkp int sym](#dynamic-shape-model-sym-bkp-int8-sym)

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Dynamic shape model sym gs128 bkp int8 sym

The workflow in Config file: [Phi-3.5-mini-instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json](Phi-3.5-mini-instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json) executes the above workflow producing a dynamic shape model.

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
olive run --config Phi-3.5-mini-instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("Phi-3.5-mini-instruct_context_ov_dynamic_sym_gs128_bkp_int8_sym.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
