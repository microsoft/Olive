# PEFT Adapaters

Parameter Efficient Finetuning (PEFT) techniques, such as LoRA enables user to efficiently finetune a model.

## LoRA
Low-Rank Adaptation, or `LoRA`, is a fine-tuning approach which freezes the pre-trained model weights and injects trainable rank decomposition matrices (called adapters) into the layers of the model.
It is based on the [LoRA paper](https://arxiv.org/abs/2106.09685).

The output model is the input transformers model along with the fine-tuned LoRA adapters. The adapters can be loaded and/or merged into the original model using the `peft` library from Hugging Face.

This pass only supports HfModels. Please refer to [LoRA](lora) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "LoRA",
    "alpha": 16,
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
    "num_epochs": 5,
    "train_data_config": "train_data_config"
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/resnet/user_script.py)
for an example implementation of `"user_script.py"` and `"train_data_config/dataloader_config/type"`.

## MergeAdapterWeights
Merge Lora weights into a complete model. After running the LoRA pass, the model will only have LoRA adapters. This pass merges the LoRA adapters into the original model and download the context(config/generation_config/tokenizer) of the model.

### Example Configuration
```json
{
    "type": "MergeAdapterWeights"
}
```

## Extract Adapters

LoRA, QLoRA and related techniques allow us to fine-tune a pre-trained model by adding a small number of trainable matrices called adapters. The same base model can be used for multiple tasks by adding different adapters for each task. To support using multiple adapters with the same optimized onnx model, the `ExtractAdapters` pass extracts the adapters weights from the model and saves them to a separate file. The model graph is then modified in one of the following ways:
- Adapters weights are set as external tensors pointing to a non-existent file. The onnx model is thus invalid by itself as it cannot be loaded. In order to create an inference session using this model, the adapter weights must be added to a sessions options object using `add_initializer` or `add_external_initializers`.
- Adapter weights are converted into model inputs. The onnx model is valid. During inference, the adapter weights must be provided as part of the inputs. We call them constant inputs here since these weights don't change between runs when using the one set of adapters.

### Example Configuration

a. As external initializers

```json
{
    "type": "ExtractAdapters",
    "make_inputs": false
}
```

b. As constant inputs with packed weights

```json
{
    "type": "ExtractAdapters",
    "make_inputs": true,
    "pack_inputs": true
}
```

Please refer to [ExtractAdapters](../../../reference/pass.rst#extract_adapters) for more details about the pass and its config parameters.

Olive also provides a command line tool to convert adapters saved after peft fine-tuning to a format compatible with a model that has been optimized with the `ExtractAdapters` pass. More details on the ``olive convert-adapters`` command can be found at [Command Line Tools](../../../reference/cli.rst).
