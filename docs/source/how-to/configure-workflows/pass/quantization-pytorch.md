# PyTorch Quantization

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

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
`QuaRot` is a quantization technique that combines quantization and rotation to reduce the number of bits required to represent the weights of a model. It is based on the [QuaRot paper](https://arxiv.org/abs/2305.14314).

This pass only supports HuggingFace transformer PyTorch models. Please refer to [QuaRot](quarot) for more details on the types of transformers models supported.

### Example Configuration
```json
{
    "type": "QuaRot",
    "w_rtn": true,
    "rotate": true,
    "w_bits": 4,
    "a_bits": 4,
    "k_bits": 4,
    "v_bits": 4
}
```
