# CLIP VIT Quantization
This folder contains examples of CLIP VIT quantization using different workflows.

- QDQ: [CLIP VIT Quantization encoded in QDQ format](#clip-vit-quantization-encoded-in-qdq-format)
- NPU: [PTQ on Qualcomm NPU using QNN EP](#ptq-on-npu)
- Intel® NPU: [Optimization with OpenVINO on Intel® NPU to generate an ONNX OpenVINO IR Encapsulated Model](./openvino/)

Go to [How to run](#how-to-run)

## Workflows

### CLIP VIT Quantization encoded in QDQ format
This workflow quantizes CLIP VIT model. It performs the pipeline:
- *PyTorch Model -> Onnx Model -> INT8 Quantized Onnx Model with QDQ format*

OpenAI clip model config file:
#### [openai_clip-vit-base-patch16_ptq_qdq.json](openai_clip-vit-base-patch16_ptq_qdq.json)

Accuracy / latency / throughput

| Model Version         | Accuracy           | Latency (ms/sample)| Throughput (token per second)| Dataset           |
|-----------------------|--------------------|--------------------|------------------------------|-------------------|
| PyTorch FP32          | 100%               | 6190.6             | 0.16                         | nlphuji/flickr30k |
| ONNX INT8 (QDQ)       | 100%               | 1525.3             | 0.65                         | nlphuji/flickr30k |


#### [openai_clip-vit-base-patch32_ptq_qdq.json](openai_clip-vit-base-patch32_ptq_qdq.json)

Accuracy / latency

| Model Version         | Accuracy           | Latency (ms/sample)  | Throughput (token per second)|
|-----------------------|--------------------|----------------------|------------------------------|
| PyTorch FP32          | 100%               | 5899.4               | 0.17                         |
| ONNX INT8 (QDQ)       | 100%               | 1117.66              | 0.96                         |

#### Open clip model config file: [laion_CLIP-ViT-B-32-laion2B-s34B-b79K_ptq_qdq.json](laion_CLIP-ViT-B-32-laion2B-s34B-b79K_ptq_qdq.json)

Accuracy / latency

| Model Version         | Accuracy          | Latency (ms/sample)| Throughput (token per second)|
|-----------------------|-------------------|--------------------|------------------------------|
| PyTorch FP32          | 100%              | 5702               | 0.17                         |
| ONNX INT8 (QDQ)       | 100%              | 1192.7             | 0.91                         |

*Note: Latency can vary significantly depending on the hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*

### PTQ on NPU
This workflow performs CLIP VIT quantization on Qualcomm NPU with ONNX Runtime PTQ. It performs the pipeline:
- *PyTorch Model -> Onnx Model -> Static shaped Onnx Model -> Quantized Onnx Model*

It requires x86 python environment on a Windows ARM machine with `onnxruntime-qnn` installed.

OpenAI clip model config file: [openai_clip-vit-base-patch16_ptq_qnn.json](openai_clip-vit-base-patch16_ptq_qnn.json)
 [openai_clip-vit-base-patch32_ptq_qnn.json](openai_clip-vit-base-patch32_ptq_qnn.json)
 [openai_clip-vit-large-patch14_ptq_qnn.json](openai_clip-vit-large-patch14_ptq_qnn.json)

Open clip model config file: [laion_CLIP-ViT-B-32-laion2B-s34B-b79K_ptq_qnn.json](laion_CLIP-ViT-B-32-laion2B-s34B-b79K_ptq_qnn.json)

**NOTE:** The model quantization part of the workflow can also be done on a Linux/Windows machine with a different onnxruntime package installed. Remove the `"evaluators"` and `"evaluator"` sections from the configuration file to skip the evaluation step.

## How to run
### Pip requirements
Install the necessary python packages:
```sh
# [NPU]
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[qnn]
```

### Other dependencies
```sh
python -m pip install -r requirements.txt
```

### Run sample using config

The quantization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.
```sh
olive run --config <config_file>.json --setup
```

Then, quantize the model
```sh
olive run --config <config_file>.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("<config_file>.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
