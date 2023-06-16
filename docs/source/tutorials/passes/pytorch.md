# PyTorch related – General

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
