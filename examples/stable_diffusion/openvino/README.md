# Stable Diffusion Optimization with OpenVINO

This sample shows how to optimize [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) to run OpenVINO IR model.

Stable Diffusion comprises multiple PyTorch models tied together into a *pipeline*. This Olive sample will convert each PyTorch model to OpenVINO IR model by `OpenVINOConversion` pass, and create an `OpenVINOStableDiffusionPipeline` for inference.

**Contents**:
- [Setup](#setup)
- [Conversion to OpenVINO IR model](#conversion-to-ov)
- [Test Inference](#test-inference)
- [Stable Diffusion Pipeline](#stable-diffusion-pipeline)

## Setup

Olive is currently under pre-release, with constant updates and improvements to the functions and usage. This sample code will be frequently updated as Olive evolves, so it is important to install Olive from source when checking out this code from the main branch. See the [README for examples](https://github.com/microsoft/Olive/blob/main/examples/README.md#important) for detailed instructions on how to do this.

**Alternatively**, you may install a stable release that we have validated. For example:

```
# Install Olive from main branch
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[openvino]

# Clone Olive repo to access sample code
git clone https://github.com/microsoft/olive
```

Once you've installed Olive, install the requirements for this sample matching the version of the library you are using:
```
cd olive/examples/stable_diffusion/openvino
pip install -r requirements.txt
```

## Convert to OpenVINO IR model

The easiest way to optimize the pipeline is with the `stable_diffusion.py` helper script:

```
python stable_diffusion.py --optimize
```

The above command will enumerate the `config_<model_name>.json` files and optimize each with Olive, then gather the optimized models into a directory structure suitable for testing inference.

The stable diffusion models are large, and the optimization process is resource intensive. It is recommended to run optimization on a system with a minimum of 16GB of memory (preferably 32GB). Expect optimization to take several minutes (especially the U-Net model).

Once the script successfully completes:
- The converted OpenVINO IR model will be stored under `models/optimized/[model_id]` (for example `models/optimized/runwayml/stable-diffusion-v1-5`).

Re-running the script with `--optimize` will delete the output models, but it will *not* delete the Olive cache. Subsequent runs will complete much faster since it will simply be copying previously optimized models; you may use the `--clean_cache` option to start from scratch (not typically used unless you are modifying the scripts, for example).

## Test Inference

This sample code is primarily intended to illustrate model optimization with Olive, but it also provides a simple interface for testing inference with the ONNX models. Inference is done by creating an `OVStableDiffusionPipeline` from the saved models.


```
python stable_diffusion.py --optimize --inference --prompt "a running dog"

Start inference with prompt: a running dog
100%|███████████████████████████████████| 20/20 [02:32<00:00,  7.63s/it]
Image saved to outputs\prompt_0.png
```

The result will be saved as `prompt_<i>.png` on disk.

Run `python stable_diffusion.py --help` for additional options. A few particularly relevant ones:
- `--model_id <string>` : name of a stable diffusion model ID hosted by huggingface.co. This script has been tested with the following:
  - `CompVis/stable-diffusion-v1-4`
  - `runwayml/stable-diffusion-v1-5` (default)
  - `prompthero/openjourney`
- `--prompt <prompt 1> <prompt 2>`: the prompt for inference. You can input multiple input by spliting with space.
- `--num_steps <int>`: the number of sampling steps per inference. The default value is 20. A lower value will speed up inference at the expensive of quality, and a higher value (e.g. 100) may produce higher quality images.
- `--image_path <str>`: the input image path for image to image inference.
- `--img_to_img_example`: image to image example. The default input image is `assets/dog.png`, the default prompt is `amazing watercolor painting`.

## Stable Diffusion Pipeline

The figure belows a high-level overview of the Stable Diffusion pipeline, and is based on a figure from [Hugging Face Blog](https://huggingface.co/blog/stable_diffusion) that covers Stable Diffusion with Diffusers library. The blue boxes are the converted & optimized ONNX models. The gray boxes remain implemented by diffusers library when using this example for inference; a custom pipeline may implement the full pipeline without leveraging Python or the diffusers library.

![sd pipeline](readme/pipeline.png)
