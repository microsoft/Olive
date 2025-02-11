# Animate Diff QNN

## Get unoptimized stable diffusion onnx model

You could either
- Download from https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX
- Get unoptmized model from `python stable_diffusion.py --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 --optimize --provider qnn` in stable_diffusion example

Put the model into models/stable-diffusion-v1-5.

## Generate unoptimized animate diff onnx model

`olive run --config config_unet.json`

Replace original unet model with the generated one (in models/stable-diffusion-v1-5/unet-animatediff).

## Test unoptmized model

`python animate_diff.py`

## Get optimized stable diffusion onnx model

Follow guide in stable_diffusion example.

Put the model into models/stable-diffusion-v1-5-qnn.

## Split unet model

The model is too big for quantization, so we need to split model first.

`python animate_diff.py --split`

## Generate data for static quantization

`python animate_diff.py --save_data --prompt "dog swims in the river"`

## Generate optimized animate diff onnx model

`python animate_diff.py --quantize --output models/stable-diffusion-v1-5-qnn`

## Test optimized model

`python animate_diff.py --input models/stable-diffusion-v1-5-qnn --output qnn.gif`
