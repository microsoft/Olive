# Stable Diffusion Optimization with DirectML

    ⚠️ THIS SAMPLE IS A WORK IN PROGRESS AND REQUIRES ONNXRUNTIME-DIRECTML 1.15+ (NOT YET RELEASED) ⚠️

This sample shows how to optimize [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) to run with ONNX Runtime and DirectML.

Stable Diffusion comprises multiple PyTorch models tied together into a *pipeline*. This Olive sample will convert each PyTorch model to ONNX, and then run the converted ONNX models through the `OrtTransformersOptimization` pass. The transformer optimization pass performs several time-consuming graph transformations that make the models more efficient for inference at runtime.

![](readme/pipeline.png)
*Based on figure from [Hugging Face Blog](https://huggingface.co/blog/stable_diffusion) that covers Stable Diffusion with Diffusers library. Blue boxes are the converted & optimized ONNX models. Gray boxes remain implemented by diffusers library.*

## Setup

Make sure that your Python environment has `onnxruntime-directml` along with other dependencies in this sample's [requirements.txt](requirements.txt):

```
pip install -r requirements.txt
```

## Conversion to ONNX and Latency Optimization

**NOTE**: the stable diffusion models are large, and the optimization process is resource intensive. We recommend running optimization on a system with a minimum of 16GB of memory (preferably 32GB).

The easiest way to optimize everything is with the `stable_diffusion.py` helper script:

```
python stable_diffusion.py --optimize
```

- The optimized models will be stored under `models/optimized/runwayml/stable-diffusion-v1-5`.
- The unoptimized models (converted to ONNX, but not run through transformer optimization pass) will be stored under `models/unoptimized/runwayml/stable-diffusion-v1-5`.
- **Alternatively**, you can optimize specific models with Olive directly. For example, `python -m olive.workflows.run --config .\config_unet.json`.

## Test Inference

The `stable_diffusion.py` helper script provides a simple wrapper around the `diffusers` library to load and execute the optimized stable diffusion models. Without any arguments, it will generate a single image with the default prompt ("a photo of an astronaut riding a horse on mars.").

```
python stable_diffusion.py
Loading models into ORT session...

Inference Batch Start (batch size = 1).
100%|█████████████████████████████| 51/51 [00:05<00:00,  9.40it/s]
Generated result_0.png
Inference Batch End (1/1 images passed the safety checker).
```

The output image will be saved to `result_0.png` in your working directory. Inference will loop until the generated image passes the safety checker (otherwise you would see black images).

You can also generate multiple images with a single session, which is more efficient than running the script repeatedly. The example below shows how to request 4 valid outputs (all using the same prompt), which will be saved as `result_0.png`, `result_1.png`, and so on. The script ran inference 6 times, because 2 of the outputs failed the safety checker.

```
python .\stable_diffusion.py --num_images 4
Loading models into ORT session...

Inference Batch Start (batch size = 1).
100%|█████████████████████████████| 51/51 [00:05<00:00,  8.92it/s]
Inference Batch End (0/1 images passed the safety checker).

Inference Batch Start (batch size = 1).
100%|█████████████████████████████| 51/51 [00:05<00:00,  9.40it/s]
Generated result_0.png
Inference Batch End (1/1 images passed the safety checker).

Inference Batch Start (batch size = 1).
100%|█████████████████████████████| 51/51 [00:05<00:00,  9.61it/s]
Generated result_1.png
Inference Batch End (1/1 images passed the safety checker).

Inference Batch Start (batch size = 1).
100%|█████████████████████████████| 51/51 [00:05<00:00,  9.68it/s]
Generated result_2.png
Inference Batch End (1/1 images passed the safety checker).

Inference Batch Start (batch size = 1).
100%|█████████████████████████████| 51/51 [00:05<00:00,  9.51it/s]
Inference Batch End (0/1 images passed the safety checker).

Inference Batch Start (batch size = 1).
100%|█████████████████████████████| 51/51 [00:05<00:00,  9.54it/s]
Generated result_3.png
Inference Batch End (1/1 images passed the safety checker).
```

Below is an example output image:

![](readme/example.png)

You can also try other prompts and adjust the number of steps in the diffusion process (default is 50 steps; fewer steps will run faster with less quality; more steps *may* have higher quality):

```
python .\stable_diffusion.py --prompt "solar eclipse, stars, realistic, space" --num_inference_steps 100

Loading models into ORT session...

Inference Batch Start (batch size = 1).
100%|█████████████████████████████| 101/101 [00:11<00:00,  9.13it/s]
Generated result_0.png
Inference Batch End (1/1 images passed the safety checker).
```

![](readme/example2.png)

# Implementation Details

The `stable_diffusion.py` helper script invokes Olive for each model independently. If you would like to convert and optimize each model by yourself, you can invoke Olive directly. For example:

```
python -m olive.workflows.run --config .\config_unet.json
```

# TODO
This sample is incomplete.

- Perform ORT runtime graph optimizations to save on session creation time. Perhaps augment `OrtPerfTuning` pass to serialize the model in ["offline mode"](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#onlineoffline-mode)? Disable graph optimizations in inference test if this is done.
- Support ORT 1.14 (need to set appropriate fusion defaults); currently only works with main branch / nightly builds.
- Consider Torch 2.0.0 support (see https://github.com/pytorch/pytorch/issues/97262).
- Investigate bland output images with batch_size > 1


# Issues

If you run into the following error while optimizing models, it is likely that your local HuggingFace cache has an incomplete copy of the stable diffusion model pipeline. Deleting `C:\users\<username>\.cache\huggingface` should resolve the issue by ensuring a fresh copy is downloaded.

```
OSError: Can't load tokenizer for 'C:\Users\<username>\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\<sha>'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'C:\Users\<username>\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\aa9ba505e1973ae5cd05f5aedd345178f52f8e6a' is the correct path to a directory containing all relevant files for a CLIPTokenizer tokenizer.
```
