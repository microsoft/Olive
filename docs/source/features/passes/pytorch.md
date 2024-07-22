# PyTorch

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

## LoRA
Low-Rank Adaptation, or `LoRA`, is a fine-tuning approach which freezes the pre-trained model weights and injects trainable rank decomposition matrices (called adapters) into the layers of the model.
It is based on the [LoRA paper](https://arxiv.org/abs/2106.09685).

The output model is the input transformers model along with the fine-tuned LoRA adapters. The adapters can be loaded and/or merged into the original model using the `peft` library from Hugging Face.

This pass only supports HfModels. Please refer to [LoRA](lora) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "LoRA",
    "lora_alpha": 16,
    "train_data_config": // ...,
    "training_args": {
        "learning_rate": 0.0002,
        // ...
    }
}
```
Please refer to [LoRA HFTrainingArguments](lora_hf_training_arguments) for more details on supported the `"training_args"` and their default values.

## QLoRA
`QLoRA` is an efficient finetuning approach that reduces memory usage by backpropagating gradients through a frozen, 4-bit quantized pretrained model into Low Rank Adapters (LoRA). It is based on
the QLoRA [paper](https://arxiv.org/abs/2305.14314) and [code](https://github.com/artidoro/qlora/blob/main/qlora.py). More information on LoRA can be found in the [paper](https://arxiv.org/abs/2106.09685).

The output model is the input transformers model along with the quantization config and the fine-tuned LoRA adapters. The adapters can be loaded and/or merged into the original model using the
`peft` library from Hugging Face.

This pass only supports HfModels. Please refer to [QLoRA](qlora) for more details about the pass and its config parameters.

**Note:** QLoRA requires a GPU to run.

### Example Configuration
```json
{
    "type": "QLoRA",
    "compute_dtype": "bfloat16",
    "quant_type": "nf4",
    "training_args": {
        "learning_rate": 0.0002,
        // ...
    },
    "train_data_config": // ...,
}
```
Please refer to [QLoRA HFTrainingArguments](lora_hf_training_arguments) for more details on supported the `"training_args"` and their default values.

## LoftQ
`LoftQ` is a quantization framework which simultaneously quantizes and finds a proper low-rank initialization for LoRA fine-tuning. It is based on the LoftQ [paper](https://arxiv.org/abs/2310.08659)
and [code](https://github.com/yxli2123/LoftQ). More information on LoRA can be found in the [paper](https://arxiv.org/abs/2106.09685).

The `LoftQ` pass initializes the quantized LoRA model using the LoftQ initialization method and then fine-tunes the adapters. The output model has new quantization aware master weights and the fine-tuned LoRA adapters.

This pass only supports HfModels. Please refer to [LoftQ](loftq) for more details about the pass and its config parameters.

**Note:** LoftQ requires a GPU to run.
```json
{
    "type": "LoftQ",
    "compute_dtype": "bfloat16",
    "training_args": {
        "learning_rate": 0.0002,
        // ...
    },
    "train_data_config": // ...,
}
```
Please refer to [LoftQ HFTrainingArguments](lora_hf_training_arguments) for more details on supported the `"training_args"` and their default values.

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
    "user_script": "user_script.py",
    "training_loop_func": "training_loop_func"
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert/user_script.py)
for an example implementation of `"user_script.py"` and `"training_loop_func"`.

b. Run QAT training with PyTorch Lightning.
```json
{
    "type": "QuantizationAwareTraining",
    "user_script": "user_script.py",
    "num_epochs": 5,
    "ptl_data_module": "PTLDataModule",
    "ptl_module": "PTLModule"
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/resnet/user_script.py)
for an example implementation of `"user_script.py"`, `"PTLDataModule"` and `"PTLModule"`.


c. Run QAT training with default training loop.
```json
{
    "type": "QuantizationAwareTraining",
    "user_script": "user_script.py",
    "num_epochs": 5,
    "train_dataloader_func": "create_train_dataloader"
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/resnet/user_script.py)
for an example implementation of `"user_script.py"` and `"create_train_dataloader"`.

## AutoGPTQ
Olive also integrates [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) for quantization.

AutoGPTQ is an easy-to-use LLM quantization package with user-friendly APIs, based on GPTQ algorithm (weight-only quantization). With GPTQ quantization, you can quantize your favorite language model to 8, 4, 3 or even 2 bits. This comes without a big drop of performance and with faster inference speed. This is supported by most GPU hardwares.

Olive consolidates the GPTQ quantization into a single pass called GptqQuantizer which supports tune GPTQ quantization with hyperparameters for trade-off between accuracy and speed.

Please refer to [GptqQuantizer](gptq_quantizer) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "GptqQuantizer",
    "data_config": "wikitext2_train"
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/llama2/llama2_template.json)
for an example implementation of `"wikitext2_train"`.

## AutoAWQ
AutoAWQ is an easy-to-use package for 4-bit quantized models and it speeds up models by 3x and reduces memory requirements by 3x compared to FP16. AutoAWQ implements the Activation-aware Weight Quantization (AWQ) algorithm for quantizing LLMs. AutoAWQ was created and improved upon from the original work from MIT.

Olive integrates [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) for quantization and make it possible to convert the AWQ quantized torch model to onnx model. You can enable `pack_model_for_onnx_conversion` to pack the model for onnx conversion.

Please refer to [AutoAWQQuantizer](awq_quantizer) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "AutoAWQQuantizer",
    "w_bit": 4,
    "pack_model_for_onnx_conversion": true
}
```

## MergeAdapterWeights
Merge Lora weights into a complete model. After running the LoRA pass, the model will only have LoRA adapters. This pass merges the LoRA adapters into the original model and download the context(config/generation_config/tokenizer) of the model.

### Example Configuration
```json
{
    "type": "MergeAdapterWeights"
}
```

## SparseGPT
`SparseGPT` prunes GPT like models using a pruning method called [SparseGPT](https://arxiv.org/abs/2301.00774). This one-shot pruning method can perform unstructured
sparsity up to 60% on large models like OPT-175B and BLOOM-176B efficiently with negligible perplexity increase. It also supports semi-structured sparsity patterns such
as 2:4 and 4:8 patterns.

Please refer to the original paper linked above for more details on the algorithm and performance results for different models, sparsities and datasets.

This pass only supports HfModels. Please refer to [SparseGPT](sparsegpt) for more details on the types of transformers models supported.

**Note:** TensorRT can accelerate inference on 2:4 sparse models as described in [this blog](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/).

### Example Configuration
```json
{
    "type": "SparseGPT",
    "sparsity": 0.5
}
```
```json
{
    "type": "SparseGPT",
    "sparsity": [2,4]
}
```

## SliceGPT
`SliceGPT` is post-training sparsification scheme that makes transformer networks smaller by applying orthogonal transformations to each transformer layer that reduces the model size by slicing off the least-significant rows and columns of the weight matrices. This results in speedups and a reduced memory footprint.

Please refer to the original [paper](https://arxiv.org/abs/2401.15024) for more details on the algorithm and expected results for different models, sparsities and datasets.

This pass only supports HuggingFace transformer PyTorch models. Please refer to [SliceGPT](slicegpt) for more details on the types of transformers models supported.

### Example Configuration
```json
{
    "type": "SliceGPT",
    "sparsity": 0.4,
    "calibration_data_config": "wikitext2"
}
```

## TorchTRTConversion
`TorchTRTConversion` converts the `torch.nn.Linear` modules in the transformer layers in a Hugging Face PyTorch model to `TRTModules` from `torch_tensorrt` with fp16 precision and sparse weights, if
applicable. `torch_tensorrt` is an extension to `torch` where TensorRT compiled engines can be used like regular `torch.nn.Module`s. This pass can be used to accelerate inference on transformer models
with sparse weights by taking advantage of the 2:4 structured sparsity pattern supported by TensorRT.

This pass only supports HfModels. Please refer to [TorchTRTConversion](torch_trt_conversion) for more details on the types of transformers models supported.

### Example Configuration
```json
{
    "type": "TorchTRTConversion"
}
```
