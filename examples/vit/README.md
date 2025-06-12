# Vision Transformer (ViT) Quantization
This folder contains examples of ViT quantization using different workflows.
- QDQ: [VIT with QDQ format](#vit-with-qdq)
- Qualcomm NPU: [Optimization with PTQ on Qualcomm NPU using QNN EP](./qnn/)
- Intel® NPU: [Optimization with OpenVINO on Intel® NPU to generate an ONNX OpenVINO IR Encapsulated Model](./openvino/)
- AMD NPU: [Optimization and Quantization with QDQ format for AMD NPU (VitisAI)](#optimization-and-quantization-for-amd-npu)
- Nvidia GPU: [With Nvidia TensorRT-RTX execution provider in ONNX Runtime](#optimization-with-nvidia-tensorrt-rtx-execution-provider)

Go to [How to run](#how-to-run)

## Workflows

### ViT with QDQ
This example performs ViT quantization in one workflow. It performs the pipeline:
- *Huggingface Model -> Onnx Model -> Quantized Onnx Model with QDQ format*

Config file: [vit_qdq.json](vit_qdq.json)

#### Accuracy / latency

| Model Version         | Accuracy            |  Latency (ms/sample) | Dataset  |
|-----------------------|---------------------|----------------------|----------|
| PyTorch FP32          | 77.3%               | 1892.2               | Imagenet |
| ONNX INT8 (QDQ)       | 77.3%               | 287.5                | Imagenet |

*Note: Latency can vary significantly depending on the CPU hardware and system environment. The values provided here are for reference only and may not reflect performance on all devices.*

### Optimization and Quantization for AMD NPU

 This workflow quantizes the model. It performs the pipeline:
 - *HF Model-> ONNX Model -> Optimizations -> Quantized Onnx Model*

 Config files for VitisAI:
 - [google/vit-base-patch16-224](vit_qdq_vitis_ai.json)

### Optimization with Nvidia TensorRT-RTX execution provider
 This example performs optimization with Nvidia TensorRT-RTX execution provider. It performs the optimization pipeline:
 - *ONNX Model -> fp16 Onnx Model*

 Config file: [vit_trtrtx.json](vit_trtrtx.json)

## How to run
```
olive run --config <config_file>.json
```

After running the above command, the final model will be saved in the *output_dir* specified in the config file.

