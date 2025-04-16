# ViT Base Patch16 224 Quantization

This folder contains a sample use case of Olive to optimize a [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) model using OpenVINO tools.

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Static shape model

The config file: [vit_base_patch16_224_context_ov_static.json](vit_base_patch16_224_context_ov_static.json) executes the above workflow producing static shape model.

## How to run

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

```bash
python -m pip install -r requirements.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.

```bash
olive run --config <config_file>.json --setup
```

Then, optimize the model

```bash
olive run --config <config_file>.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("<config_file>.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
