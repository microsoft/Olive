# CLIP VIT Quantization
This folder contains examples of CLIP VIT quantization using different workflows.

- QDQ: [CLIP VIT Quantization encoded in QDQ format](#clip-vit-quantization-encoded-in-qdq-format)
- Qualcomm NPU: [Optimization with PTQ on Qualcomm NPU using QNN EP](./qnn/)
- Intel® NPU: [Optimization with OpenVINO on Intel® NPU to generate an ONNX OpenVINO IR Encapsulated Model](./openvino/)
- AMD NPU: [Optimization and Quantization with QDQ format for AMD NPU (VitisAI)](#optimization-and-quantization-for-amd-npu)

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

### Optimization and Quantization for AMD NPU

 This workflow quantizes the model. It performs the pipeline:
 - *HF Model-> ONNX Model -> Optimizations -> Quantized Onnx Model*

 Config files for VitisAI:
 - [laion/CLIP-ViT-B-32-laion2B-s34B-b79K](laion_CLIP-ViT-B-32-laion2B-s34B-b79K_ptq_qdq_vitis_ai.json)
 - [openai/clip-vit-base-patch16](openai_clip-vit-base-patch16_ptq_qdq_vitis_ai.json)
 - [openai/clip-vit-base-patch32](openai_clip-vit-base-patch32_ptq_qdq_vitis_ai.json)

## How to run
### Pip requirements
Install the necessary python packages:
```sh
# [NPU]
pip install git+https://github.com/microsoft/Olive#egg=olive-ai
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

After running the above command, the final model will be saved in the *output_dir* specified in the config file.
