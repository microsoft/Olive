# Vision Transformer (ViT) Optimization
This folder contains examples of ViT optimization using different workflows.
- Qualcomm NPU: [with QNN execution provider in ONNX Runtime](#vit-optimization-with-qnn-execution-providers)

## Optimization Workflows
### ViT optimization with QNN execution providers
This example performs ViT optimization with QNN execution providers in one workflow. It performs the optimization pipeline:
- *Huggingface Model -> Onnx Model -> QNN Quantized Onnx Model*

Config file: [vit_qnn_config.json](vit_qnn_config.json)
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
