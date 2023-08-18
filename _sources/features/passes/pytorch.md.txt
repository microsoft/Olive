# PyTorch

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

## Quantization Aware Training
The Quantization Aware Training (QAT) technique is used to improve the performance and efficiency of deep learning models by quantizing their
weights and activations to lower bit-widths. The technique is applied during training, where the weights and activations are fake quantized
to lower bit-widths using the specified QConfig.

Olive provides `QuantizationAwareTraining` that performs QAT on a PyTorch model.

Please refer to [QuantizationAwareTraining](quantization_aware_training) for more details about the pass and its config parameters.

### Example Configuration
Olive provides the 3 ways to run QAT training process:

a. Run QAT training with customized training loop.
```json
{
    "type": "QuantizationAwareTraining",
    "config":{
        "user_script": "user_script.py",
        "training_loop_func": "training_loop_func"
    }
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert/user_script.py)
for an example implementation of `"user_script.py"` and `"training_loop_func"`.

b. Run QAT training with PyTorch Lightning.
```json
{
    "type": "QuantizationAwareTraining",
    "config":{
        "user_script": "user_script.py",
        "num_epochs": 5,
        "ptl_data_module": "PTLDataModule",
        "ptl_module": "PTLModule",
    }
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/resnet/user_script.py)
for an example implementation of `"user_script.py"`, `"PTLDataModule"` and `"PTLModule"`.


c. Run QAT training with default training loop.
```json
{
    "type": "QuantizationAwareTraining",
    "config":{
        "user_script": "user_script.py",
        "num_epochs": 5,
        "train_dataloader_func": "create_train_dataloader",
    }
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/resnet/user_script.py)
for an example implementation of `"user_script.py"` and `"create_train_dataloader"`.

## SparseGPT
`SparseGPT` prunes GPT like models using a pruning method called [SparseGPT](https://arxiv.org/abs/2301.00774). This one-shot pruning method can perform unstructured
sparsity upto 60% on large models like OPT-175B and BLOOM-176B efficiently with negligible perplexity increase. It also supports semi-structured sparsity patterns such
as 2:4 and 4:8 patterns.

Please refer to the original paper linked above for more details on the algorithm and performance results for different models, sparsities and datasets.

This pass only supports Hugging Face transformers PyTorch models. Please refer to [SparseGPT](sparsegpt) for more details on the types of transformers models supported.

**Note:** TensorRT can accelerate inference on 2:4 sparse models as described in [this blog](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/).

### Example Configuration
```json
{
    "type": "SparseGPT",
    "config": {"sparsity": 0.5}
}
```
```json
{
    "type": "SparseGPT",
    "config": {"sparsity": [2,4]}
}
```

## TorchTRTConversion
`TorchTRTConversion` converts the `torch.nn.Linear` modules in the transformer layers in a Hugging Face PyTorch model to `TRTModules` from `torch_tensorrt` with fp16 precision and sparse weights, if
applicable. `torch_tensorrt` is an extension to `torch` where TensorRT compiled engines can be used like regular `torch.nn.Module`s. This pass can be used to accelerate inference on transformer models
with sparse weights by taking advantage of the 2:4 structured sparsity pattern supported by TensorRT.

This pass only supports Hugging Face transformers PyTorch models. Please refer to [TorchTRTConversion](torch_trt_conversion) for more details on the types of transformers models supported.

### Example Configuration
```json
{
    "type": "TorchTRTConversion"
}
```
