# Vision Transformer (ViT) Quantization
This folder contains examples of ViT quantization using different workflows.
- QDQ: [VIT with QDQ format](#vit-with-qdq)
- Qualcomm NPU: [with QNN execution provider in ONNX Runtime](#vit-quantization-with-qnn-execution-providers)
- Intel® NPU: [Optimization with OpenVINO on Intel® NPU to generate an ONNX OpenVINO IR Encapsulated Model](./openvino/)

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

### ViT quantization with QNN execution providers
This example performs ViT quantization with QNN execution providers in one workflow. It performs the pipeline:
- *Huggingface Model -> Onnx Model -> QNN Quantized Onnx Model*

Config file: [vit_qnn.json](vit_qnn.json)

## How to run
```
olive run --config <config_file>.json
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.

### Test with Tiny-ImageNet-200
Tiny-ImageNet-200 is a smaller subset of the ImageNet dataset containing 200 classes, commonly used for benchmarking deep learning models.

You can test output model with provided scripts. It is also a example you can refer about inference with model.
- Download dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip and extract.
- Go to subfolder *val_tiny_imagenet*. In *val_tiny_imagenet.py*, update *path_to_tiny_imagenet* with Tiny-ImageNet-200 root path and *path_to_model*. Modify *limit* as how many number you want in your test.
- Run
```
python .\val_tiny_imagenet.py
```
