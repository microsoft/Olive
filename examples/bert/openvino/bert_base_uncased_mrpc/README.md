# BERT Base Uncased MRPC Quantization

This folder contains a sample use case of Olive to optimize a [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) model using OpenVINO tools.

- Intel® NPU: [BERT Base Uncased MRPC static shape model](#static-shape-model)

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Static shape model

The config file: [bert-base-uncased-mrpc_context_ov_static.json](bert-base-uncased-mrpc_context_ov_static.json) executes the above workflow producing static shape model.

## How to run

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

```bash
olive run --config bert-base-uncased-mrpc_context_ov_static.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("bert-base-uncased-mrpc_context_ov_static.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
