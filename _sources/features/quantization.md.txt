# Quantization

## AutoGPTQ
Olive integrates [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) for quantization.

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

Olive integrates [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) for quantization and make it possible to convert the AWQ quantized torch model to onnx model.

Please refer to [AutoAWQQuantizer](awq_quantizer) for more details about the pass and its config parameters.

### Example Configuration
```json
{
    "type": "AutoAWQQuantizer",
    "w_bit": 4
}
```

## QuaRot
`QuaRot` is a technique that rotates the weights of a model to make them more conducive to quantization. It is based on the [QuaRot paper](https://arxiv.org/abs/2305.14314) but only performs offline weight rotation. Can be followed by a pass such as GPTQ to quantize the rotated model weights.

This pass only supports HuggingFace transformer PyTorch models.

### Example Configuration
```json
{
    "type": "QuaRot",
    "rotate_mode": "hadamard"
}
```

## SpinQuant
`SpinQuant` is a technique simlar to QuaRot that rotates the weights of a model to make them more conducive to quantization. The rotation weights are trained on a calibration dataset to improve activation quantization quality. It is based on the [SpinQuant paper](https://arxiv.org/pdf/2405.16406) but only performs offline weight rotation. Can be followed by a pass such as GPTQ to quantize the rotated model weights.

This pass only supports HuggingFace transformer PyTorch models.

### Example Configuration
```json
{
    "type": "SpinQuant",
    "rotate_mode": "hadamard",
    "a_bits": 8
}
```

## Quantize with onnxruntime
Quantization is a technique to compress deep learning models by reducing the precision of the model weights from 32 bits to 8 bits. This
technique is used to reduce the memory footprint and improve the inference performance of the model. Quantization can be applied to the
weights of the model, the activations of the model, or both.

There are two ways to quantize a model in onnxruntime:

1. *Dynamic Quantization*: Dynamic quantization calculates the quantization parameters (scale and zero point) for activations dynamically, which means there is no
    any requirement for the calibration dataset. These calculations increase the cost of inference, while usually achieve higher accuracy comparing to static ones.
2. *Static Quantization*:  Static quantization method runs the model using a set of inputs called calibration data. In this way, user must provide a calibration
    dataset to calculate the quantization parameters (scale and zero point) for activations before quantizing the model.

Olive consolidates the dynamic and static quantization into a single pass called `OnnxQuantization`, and provide the user with the ability to
tune both quantization methods and hyperparameter at the same time.
If the user desires to only tune either of dynamic or static quantization, Olive also supports them through `OnnxDynamicQuantization` and
`OnnxStaticQuantization` respectively.

Please refer to [OnnxQuantization](onnx_quantization), [OnnxDynamicQuantization](onnx_dynamic_quantization) and
[OnnxStaticQuantization](onnx_static_quantization) for more details about the passes and their config parameters.

**Note:** If target execution provider is QNN EP, the model might need to be preprocessed before quantization. Please refer to [QnnPreprocess](qnn_preprocess) for more details about the pass and its config parameters.
This preprocessing step fuses operators unsupported by QNN EP and inserts necessary operators to make the model compatible with QNN EP.

### Example Configuration
a. Tune the parameters of the OlivePass with pre-defined searchable values
```json
{
    "type": "OnnxQuantization",
    "data_config": "calib_data_config"
}
```

b. Select parameters to tune
```json
{
    "type": "OnnxQuantization",
    // select per_channel to tune with "SEARCHABLE_VALUES".
    // other parameters will use the default value, not to be tuned.
    "per_channel": "SEARCHABLE_VALUES",
    "data_config": "calib_data_config"
}
```

c. Use default values of the OlivePass (no tuning in this way)
```json
{
    "type": "OnnxQuantization",
    // set per_channel to "DEFAULT_VALUE"
    "per_channel": "DEFAULT_VALUE",
    "data_config": "calib_data_config"
}
```

d. Specify parameters with user defined values
```json
"onnx_quantization": {
    "type": "OnnxQuantization",
    // set per_channel to True.
    "per_channel": true,
    "data_config": "calib_data_config"
}
```

Check out [this file](https://github.com/microsoft/Olive/blob/main/examples/bert/user_script.py)
for an example implementation of `"user_script.py"` and `"calib_data_config/dataloader_config/type"`.

check out [this file](https://github.com/microsoft/Olive/tree/main/examples/bert#bert-optimization-with-intel-neural-compressor-ptq-on-cpu) for an example for Intel® Neural Compressor quantization.

## Quantize with Intel® Neural Compressor
In addition to the default onnxruntime quantization tool, Olive also integrates [Intel® Neural Compressor](https://github.com/intel/neural-compressor).

Intel® Neural Compressor is a model compression tool across popular deep learning frameworks including TensorFlow, PyTorch, ONNX Runtime (ORT) and MXNet, which supports a variety of powerful model compression techniques, e.g., quantization, pruning, distillation, etc. As a user-experience-driven and hardware friendly tool, Intel® Neural Compressor focuses on providing users with an easy-to-use interface and strives to reach “quantize once, run everywhere” goal.

Olive consolidates the Intel® Neural Compressor dynamic and static quantization into a single pass called `IncQuantization`, and provide the user with the ability to
tune both quantization methods and hyperparameter at the same time.
If the user desires to only tune either of dynamic or static quantization, Olive also supports them through `IncDynamicQuantization` and
`IncStaticQuantization` respectively.

### Example Configuration
```json
"inc_quantization": {
    "type": "IncStaticQuantization",
    "approach": "weight_only",
    "weight_only_config": {
        "bits": 4,
        "algorithm": "GPTQ"
    },
    "data_config": "calib_data_config",
    "calibration_sampling_size": [8],
    "save_as_external_data": true,
    "all_tensors_to_one_file": true
}
```

Please refer to [IncQuantization](inc_quantization), [IncDynamicQuantization](inc_dynamic_quantization) and
[IncStaticQuantization](inc_static_quantization) for more details about the passes and their config parameters.

## Quantize with AMD Vitis AI Quantizer
Olive also integrates [AMD Vitis AI Quantizer](https://github.com/microsoft/Olive/blob/main/olive/passes/onnx/vitis_ai/quantize.py) for quantization.

The Vitis™ AI development environment accelerates AI inference on AMD® hardware platforms. The Vitis AI quantizer can reduce the computing complexity by converting the 32-bit floating-point weights and activations to fixed-point like INT8. The fixed-point network model requires less memory bandwidth, thus providing faster speed and higher power efficiency than the floating-point model.
Olive consolidates the Vitis™ AI quantization into a single pass called VitisAIQuantization which supports power-of-2 scale quantization methods and supports Vitis AI Execution Provider.

### Example Configuration
```json
"vitis_ai_quantization": {
    "type": "VitisAIQuantization",
    "calibrate_method":"NonOverflow",
    "quant_format":"QDQ",
    "activation_type":"QUInt8",
    "weight_type":"QInt8",
    "data_config": "calib_data_config"
}
```
Please refer to [VitisAIQuantization](vitis_ai_quantization) for more details about the pass and its config parameters.

## NVIDIA TensorRT Model Optimizer-Windows
Olive also integrates [TensorRT Model Optimizer-Windows](https://github.com/NVIDIA/TensorRT-Model-Optimizer)

The TensorRT Model Optimizer-Windows is engineered to deliver advanced model compression techniques, including quantization, to Windows RTX PC systems. Specifically tailored to meet the needs of Windows users,it is optimized for rapid and efficient quantization, featuring local GPU calibration, reduced system and video memory consumption, and swift processing times.

The primary objective of the TensorRT Model Optimizer-Windows is to generate optimized, standards-compliant ONNX-format models for DirectML backends. This makes it an ideal solution for seamless integration with ONNX Runtime (ORT) and DirectML (DML) frameworks, ensuring broad compatibility with any inference framework supporting the ONNX standard.

Olive consolidates the NVIDIA TensorRT Model Optimizer-Windows quantization into a single pass called NVModelOptQuantization which supports AWQ algorithm.

### Example Configuration
```json
"quantization": {
    "type": "NVModelOptQuantization",
    "algorithm": "AWQ",
    "tokenizer_dir": "microsoft/Phi-3-mini-4k-instruct",
    "calibration": "awq_lite"
}
```

Please refer to [Phi3 example](https://github.com/microsoft/Olive/tree/main/examples/phi3#quantize-using-nvidia-tensorrt-model-optimizer)  for usability and setup details.

