# DeepSeek-R1-Distill-Qwen-1.5B Quantization

This folder contains a sample use case of Olive to optimize a [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) model using OpenVINO tools.

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Dynamic shape model

The workflow in Config file: [DeepSeek-R1-Distill-Qwen-1.5B_context_ov_dynamic_sym_gs128_bkp_int8_sym_r1.json](DeepSeek-R1-Distill-Qwen-1.5B_context_ov_dynamic_sym_gs128_bkp_int8_sym_r1.json) executes the above workflow producing a dynamic shape model.

## How to run

### Pip requirements

Install the necessary python packages:

```bash
python -m pip install -r requirements.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model

```bash
olive run --config DeepSeek-R1-Distill-Qwen-1.5B_context_ov_dynamic_sym_gs128_bkp_int8_sym_r1.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("DeepSeek-R1-Distill-Qwen-1.5B_context_ov_dynamic_sym_gs128_bkp_int8_sym_r1.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
