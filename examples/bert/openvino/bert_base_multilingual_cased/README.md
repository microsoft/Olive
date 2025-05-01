# BERT Base Multilingual Cased Quantization

This folder contains a sample use case of Olive to optimize a [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) model using OpenVINO tools.

- Intel® NPU: [BERT Base Multilingual Cased static shape model](#static-shape-model)

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Static shape model

The config file: [bert-base-multilingual-cased_context_ov_static.json](bert-base-multilingual-cased_context_ov_static.json) executes the above workflow producing static shape model.

## How to run

Install the necessary python packages:

```bash
python -m pip install olive-ai[openvino]
python -m pip install -r requirements.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

```bash
olive run --config bert-base-multilingual-cased_context_ov_static.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("bert-base-multilingual-cased_context_ov_static.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
