# Stable Diffusion Optimization with DirectML

This sample shows how to optimize [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) to run with ONNX Runtime and DirectML.


## Setup

This sample uses pre-trained models hosted by [Hugging Face](https://huggingface.co/), for which you need an account. Once you've set up an account, generate an [access token](https://huggingface.co/docs/hub/security-tokens) and log in with a terminal:

```
huggingface-cli.exe login
```

The above command will ask for your access token, which you can find on your account profile `Settings -> Access Tokens`, just copy it from here and carefully paste it on this prompt. Note that you won't see anything appear on the prompt when you paste it, that's fine. It's there already, just hit Enter. You'll start downloading the model from Hugging Face.

Next, make sure that your Python environment has `onnxruntime-directml` along with other dependencies in this sample's [requirements.txt](requirements.txt):

```
pip install -r requirements.txt
```

## Conversion to ONNX and Latency Optimization

Stable Diffusion comprises multiple PyTorch models glued together into a *pipeline*. Each model is converted to ONNX, and then each ONNX model is optimized using the `OrtStableDiffusionOptimization` pass. The optimization pass performs several time-consuming graph transformations that make the model more efficient for inference with DirectML. The easiest way to optimize everything is with the `stable_diffusion.py` helper script:

```
python stable_diffusion.py --optimize
```

The optimized models will be stored under `models/runwayml/stable-diffusion-v1-5`.

## Test Inference

The `stable_diffusion.py` helper script also provides a simple wrapper around the `diffusers` library to load and execute the optimized stable diffusion models.

```
python stable_diffusion.py --prompt "frog wearing a pirate hat, painting"
```

The output image will be saved to `result.png` in your working directory. Inference will loop until the generated image passes the safety checker (otherwise you would see black images).

![example output](readme/example.png)

# Implementation Details

The `stable_diffusion.py` helper script invokes Olive for each model independently. If you would like to convert and optimize each model by yourself, you can invoke Olive directly. For example:

```
python -m olive.workflows.run --config .\config_unet.json
```

# TODO
This sample is incomplete.

- Currently assumes FP16 conversion is always done. There should be an option to use FP32 and copy the external weights.
- Perform ORT runtime graph optimizations to save on session creation time. Perhaps augment `OrtPerfTuning` pass to serialize the model in ["offline mode"](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode)? Disable graph optimizations in inference test if this is done.
- Support ORT 1.14 (need to set appropriate fusion defaults); currently only works with main branch / nightly builds.
- Consider Torch 2.0.0 support (see https://github.com/pytorch/pytorch/issues/97262).