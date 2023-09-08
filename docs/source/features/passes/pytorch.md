# PyTorch

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

## QLoRA
`QLoRA` is an efficient finetuning approach that reduces memory usage by backpropagating gradients through a frozen, 4-bit quantized pretrained model into Low Rank Adapters (LoRA). It is based on
the QLoRA [paper](https://arxiv.org/abs/2305.14314) and [code](https://github.com/artidoro/qlora/blob/main/qlora.py). More information on LoRA can be found in the [paper](https://arxiv.org/abs/2106.09685).

The output model is the input transformers model along with the quantization config and the fine-tuned LoRA adapters. The adapters can be loaded and/or merged into the original model using the
`peft` library from Hugging Face.

This pass only supports Hugging Face transformers PyTorch models. Please refer to [QLoRA](qlora) for more details about the pass and its config parameters.

**Note:** QLoRA requires a GPU to run.

### Example Configuration
```json
{
    "type": "QLoRA",
    "config": {
        "compute_dtype": "bfloat16",
        "quant_type": "nf4",
        "train_data_config": // ...,
        "training_args": {
            "learning_rate": 0.0002,
            // ...
        }
    }
}
```
Please refer to [QLoRA HFTrainingArguments](qlora_hf_training_arguments) for more details on supported the `"training_args"` and their default values.

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
