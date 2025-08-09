# ResNet50 Quantization

This folder contains a sample use case of Olive to optimize a [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) model using OpenVINO tools.

- Intel® NPU OpenVINO Model: [Resnet50 static shape ONNX encapsulated OpenVINO IR model](#openvino-static-shape-model)
- Intel® NPU ONNX Model: [Resnet50 static shape ONNX model](#onnx-static-shape-model)

## OpenVINO Quantization Workflows with OpenVINO NNCF

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> OpenVINO Model -> Quantized OpenVINO Model -> Quantized encapsulated ONNX OpenVINO IR Model*

### OpenVINO static shape model

The config file: [resnet_context_ov_static.json](resnet_context_ov_static.json) executes the above workflow producing static shape quantized encapsulated ONNX OpenVINO IR model.

## ONNX Model Quantization Workflows with OpenVINO NNCF

This workflow performs quantization with OpenVINO NNCF. It performs the optimization pipeline:

- *HuggingFace Model -> ONNX Model -> Quantized ONNX Model*

### ONNX static shape model

The config file: [resnet_onnx_static.json](resnet_onnx_static.json) executes the above workflow producing static shape quantized ONNX model.

## How to run

### Pip requirements

Install the necessary python packages:

```bash
python -m pip install -r requirements.txt
```

**NOTE:**

- Access to the [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset is gated and therefore you will need to request access to the dataset. Once you have access to the dataset, you'll need to log-in to Hugging Face with a [user access token](https://huggingface.co/docs/hub/security-tokens) so that Olive can download it.

```bash
huggingface-cli login
```

### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

Optimize the model using the following command

```bash
olive run --config <config_file.json>
```

Example:

```bash
olive run --config resnet_context_ov_static.json
```

or run simply with python code:

```python
from olive import run
workflow_output = run("<config_file.json>")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
