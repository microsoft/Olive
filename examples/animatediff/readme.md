# Animate Diff QNN

## Get unoptimized stable diffusion onnx model

You could either
- Download from https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX
- Get unoptmized model from `python stable_diffusion.py --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 --optimize --provider qnn` in stable_diffusion example

## Generate unoptimized animate diff onnx model

`olive run --config config_unet.json`

Replace original unet model with the generated one.

## Test unoptmized model

`python OnnxAnimateDiffPipeline.py --seed 0`

## Get optimized stable diffusion onnx model

Follow guide in stable_diffusion example.

## Generate data for static quantization

`python OnnxAnimateDiffPipeline.py --seed 0 --save_data --prompt "dog swims in the river"`

## Generate optimized animate diff onnx model

`olive run --config config_unet_qnn.json`

## Test optimized model

`python OnnxAnimateDiffPipeline.py --seed 0 --input models/stable-diffusion-v1-5-qnn`
