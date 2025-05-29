# Vision Transformer (ViT) Optimization with PTQ on Qualcomm NPU using QNN EP
This example performs ViT optimization on Qualcomm NPU with ONNX Runtime PTQ. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Quantized Onnx Model*

It requires x86 python environment on a Windows ARM machine with `onnxruntime-qnn` installed.

**NOTE:** The model quantization part of the workflow can also be done on a Linux/Windows machine with a different onnxruntime package installed. Remove the `"evaluators"` and `"evaluator"` sections from the configuration file to skip the evaluation step.

## Test with Tiny-ImageNet-200
Tiny-ImageNet-200 is a smaller subset of the ImageNet dataset containing 200 classes, commonly used for benchmarking deep learning models.

You can test output model with provided scripts. It is also a example you can refer about inference with model.
- Download dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip and extract.
- Go to subfolder *val_tiny_imagenet*. In *val_tiny_imagenet.py*, update *path_to_tiny_imagenet* with Tiny-ImageNet-200 root path and *path_to_model*. Modify *limit* as how many number you want in your test.
- Run
```
python .\val_tiny_imagenet.py
```
