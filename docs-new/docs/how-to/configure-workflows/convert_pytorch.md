# PyTorch

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

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
