# ResNet50 Quantization

This folder contains a sample use case of Olive to optimize a [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) model using OpenVINO tools.

## Quantization Workflows

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

### Static shape model

The config file: [resnet_context_ov_static.json](resnet_context_ov_static.json) executes the above workflow producing static shape model.

## How to run

### Pip requirements
Install the necessary python packages:

```bash
python -m pip install -r requirements.txt
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.

```bash
olive run --config resnet_context_ov_static.json --setup
```

Then, optimize the model

```bash
olive run --config resnet_context_ov_static.json
```

or run simply with python code:

```python
from olive.workflows import run as olive_run
olive_run("resnet_context_ov_static.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
