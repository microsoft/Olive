# Stable Diffusion Optimization

This folder contains sample use cases of Olive with ONNX Runtime and OpenVINO to optimize:
- Stable Diffusion: [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2)

Stable Diffusion comprises multiple PyTorch models tied together into a *pipeline*.

The ONNX Runtime optimization sample will convert each PyTorch model to ONNX, and then run the converted ONNX models through the `OrtTransformersOptimization` pass. The transformer optimization pass performs several time-consuming graph transformations that make the models more efficient for inference at runtime.

The OpenVINO optimization sample will convert each PyTorch model to OpenVINO IR model by `OpenVINOConversion` pass, and create an `OpenVINOStableDiffusionPipeline` for inference.

- ONNX Runtime with
    - [CUDA EP](#stable-diffusion-optimization-with-onnx-runtime-cuda-ep)
    - DirectML EP: go to examples [Stable Diffusion](../directml/stable_diffusion/README.md)
- [OpenVINO](#stable-diffusion-optimization-with-openvino)

## Stable Diffusion Optimization with ONNX Runtime CUDA EP

This sample performs the following optimization workflow for each model in the Stable Diffusion pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model fp16*
<br/><br/>

Transformers optimization uses the following optimizations to speed up Stable Diffusion in CUDA:
* [Flash Attention](https://arxiv.org/abs/2205.14135) for float16 precision. Flash Attention uses tiling to reduce number of GPU memory reads/writes, and improves performance with less memory for long sequence length. The kernel requires GPUs of Compute Capability >= 7.5 (like T4, A100, and RTX 2060~4090). Only availanle in Linux.
* [Memory Efficient Attention](https://arxiv.org/abs/2112.05682v2) for float32 precision or older GPUs (like V100). We used the fused multi-head attention kernel in CUTLASS, and the kernel was contributed by xFormers.
* Channel-last (NHWC) convolution. For NVidia GPU with Tensor Cores support, NHWC tensor layout is recommended for convolution. See [Tensor Layouts In Memory: NCHW vs NHWC](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout).
* GroupNorm for NHWC tensor layout, and SkipGroupNorm fusion which fuses GroupNorm with Add bias and residual inputs
* SkipLayerNormalization which fuses LayerNormalization with Add bias and residual inputs.
* BiasSplitGelu is a fusion of Add bias with SplitGelu activation.
* BiasAdd fuses Add bias and residual.
* Reduce Transpose nodes by graph transformation.

#### Prerequisites
##### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.


We use the same olive workflow config files and scripts as the DirectML examples. The only difference is the `--provider cuda` option provided to the `stable_diffusion.py` scripts.

So, cd into the corresponding DirectML example folder from the root of the cloned repository:

**_Stable Diffusion_**
```bash
cd examples/stable_diffusion
```

##### Install onnxruntime

This example requires the latest onnxruntime-gpu code which can either be built from source or installed from the nightly builds. The following command can be used to install the latest nightly build of onnxruntime-gpu:

```bash
# uninstall any pre-existing onnxruntime packages
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml ort-nightly ort-nightly-gpu ort-nightly-directml

# install onnxruntime-gpu nightly build
pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
pip install onnxruntime-gpu --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

##### Install other dependencies

Install the necessary python packages:

```bash
python -m pip install -r requirements-common.txt
```

#### Conversion to ONNX and Latency Optimization

The easiest way to optimize the pipeline is with the `stable_diffusion.py` scripts. These scripts will enumerate the `config_<model_name>.json` files and optimize each with Olive, then gather the optimized models into a directory structure suitable for testing inference.

**_Stable Diffusion_**
```bash
# default model_id is "CompVis/stable-diffusion-v1-4"
python stable_diffusion.py --provider cuda --optimize
```

Once the script successfully completes:
- The optimized ONNX pipeline will be stored under `models/optimized-cuda/[model_id]` (for example `models/optimized-cuda/CompVis/stable-diffusion-v1-4`).
- The unoptimized ONNX pipeline (models converted to ONNX, but not run through transformer optimization pass) will be stored under `models/unoptimized/[model_id]` (for example `models/unoptimized/CompVis/stable-diffusion-v1-4`).

Re-running the script with `--optimize` will delete the output models, but it will *not* delete the Olive cache. Subsequent runs will complete much faster since it will simply be copying previously optimized models; you may use the `--clean_cache` option to start from scratch (not typically used unless you are modifying the scripts, for example).

### Test Inference with CUDA

Test ONNX runtime inference with the optimized models using `OnnxStableDiffusionPipeline`:

**_Stable Diffusion_**
```bash
python stable_diffusion.py --provider cuda --num_images 2
```
Inference will loop until the generated image passes the safety checker (otherwise you would see black images). The result will be saved as `result_<i>.png` on disk.

Refer to the corresponding section in the DirectML READMEs for more details on the test inference options:
- [Stable Diffusion](../directml/stable_diffusion/README.md#test-inference)

## Stable Diffusion Optimization with OpenVINO

**Contents**:
- [Setup](#setup)
- [Conversion to OpenVINO IR model](#convert-to-openvino-ir-model)
- [Test Inference](#test-inference-with-openvino)

### Setup

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
cd olive/examples/stable_diffusion
pip install -r requirements-ov.txt
```

##### Install onnxruntime

This example requires onnxruntime and its openvino module to be installed:

```bash
pip install onnxruntime-openvino
```

### Convert to OpenVINO IR model

The easiest way to optimize the pipeline is with the `stable_diffusion.py` helper script:

```
python stable_diffusion.py --optimize
```

The above command will enumerate the `config_<model_name>.json` files and optimize each with Olive, then gather the optimized models into a directory structure suitable for testing inference.

The stable diffusion models are large, and the optimization process is resource intensive. It is recommended to run optimization on a system with a minimum of 16GB of memory (preferably 32GB). Expect optimization to take several minutes (especially the U-Net model).

Once the script successfully completes:
- The converted OpenVINO IR model will be stored under `models/optimized-openvino/[model_id]` (for example `models/optimized-openvino/CompVis/stable-diffusion-v1-4`).

Re-running the script with `--optimize` will delete the output models, but it will *not* delete the Olive cache. Subsequent runs will complete much faster since it will simply be copying previously optimized models; you may use the `--clean_cache` option to start from scratch (not typically used unless you are modifying the scripts, for example).

### Test Inference with OpenVINO

This sample code is primarily intended to illustrate model optimization with Olive, but it also provides a simple interface for testing inference with the OpenVINO models. Inference is done by creating an `OVStableDiffusionPipeline` from the saved models.


```
python stable_diffusion.py --inference --provider openvino
```
Inference will loop until the generated image. The result will be saved as `result_<i>.png` on disk.


Run `python stable_diffusion.py --help` for additional options. A few particularly relevant ones:
- `--image_path <str>`: the input image path for image to image inference.
- `--img_to_img_example`: image to image example. The default input image is `assets/dog.png`, the default prompt is `amazing watercolor painting`.

## Stable Diffusion Quantization encoded in QDQ format

### Generate data for static quantization

To get better result, we need to generate real data from original model instead of using random data for static quantization.

First generate onnx unoptimized model:

`python stable_diffusion.py --model_id stabilityai/sd-turbo --provider cpu --format qdq --optimize --only_conversion`

Then generate data:

`python .\evaluation.py --save_data --model_id stabilityai/sd-turbo --num_inference_steps 1 --seed 0 --num_data 100 --guidance_scale 0`

### Optimize

`python stable_diffusion.py --model_id stabilityai/sd-turbo --provider cpu --format qdq --optimize`

### Test and evaluate

`python .\evaluation.py --model_id stabilityai/sd-turbo --num_inference_steps 1 --seed 0 --num_data 100 --guidance_scale 0`

To generate one image:

`python stable_diffusion.py --model_id stabilityai/sd-turbo --provider cpu --format qdq --guidance_scale 0 --seed 0 --num_inference_steps 1 --prompt "A baby is laying down with a teddy bear"`
