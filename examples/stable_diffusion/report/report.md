# Static Quantize Stable Diffusion via Olive Report

## Data and Config

These are the parameters I used to generate the quantization data and evaluate the original and generated model

- Model: stabilityai/stable-diffusion-2-1
- Num steps: 10
- Guidance Scale: 7.5
- Prompts: Use 10 captions from https://huggingface.co/datasets/laion/relaion2B-en-research-safe, 8 for training and 2 for testing

## Quantization steps

For all text encoder, unet and vae decoder model, I use the following passes from Olive

- https://microsoft.github.io/Olive/reference/pass.html#onnxpeepholeoptimizer
- https://microsoft.github.io/Olive/reference/pass.html#qnnpreprocess
    + with fuse_layernorm = true
- https://microsoft.github.io/Olive/reference/pass.html#onnxstaticquantization
    + with quant_preprocess = true, prepare_qnn_config = true, activation_type = QUInt16 and weight_type = QUInt8
    + by default, it uses QDQ quant_format and MinMax as calibrate_method

For text encoder, I didn't quantize Add and Softmax nodes. For unet and vae decoder, all nodes are quantized.

## Result

For a clear comparison, only one model uses quantized version each time.

The images for original model are in (unoptimized)[./unoptimized].

### text encoder is quantized


