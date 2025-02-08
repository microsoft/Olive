# Get unoptimized stable diffusion onnx model

You could either
- Download from https://huggingface.co/onnx-community/stable-diffusion-v1-5-ONNX
- Get unoptmized model from `python stable_diffusion.py --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 --optimize` in stable_diffusion example

# Generate unoptimized animate diff onnx model

`olive run --config config_unet.json`

Replace original unet model with the generated one.

# Test unoptmized model

`python OnnxAnimateDiffPipeline.py --seed 0`
