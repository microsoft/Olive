# Stable Diffusion and Stable Diffusion XL Optimization

This folder contains sample use cases of Olive to optimize:
- Stable Diffusion: [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Stable Diffusion v2](https://huggingface.co/stabilityai/stable-diffusion-2)
- Stable Diffusion XL: [Stable Diffusion XL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [Stable Diffusion XL Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

Stable Diffusion comprises multiple PyTorch models tied together into a *pipeline*. This Olive sample will convert each PyTorch model to ONNX, and then run the converted ONNX models through the `OrtTransformersOptimization` pass. The transformer optimization pass performs several time-consuming graph transformations that make the models more efficient for inference at runtime.

See the following for instructions on how to optimize Stable Diffusion models with Olive for different ONNX Runtime execution providers:
- DirectML:
    - [Stable Diffusion](../directml/stable_diffusion/README.md)
    - [Stable Diffusion XL](../directml/stable_diffusion_xl/README.md)
- [CUDA](#optimization-with-cuda)

## Optimization with CUDA
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

### Prerequisites
#### Clone the repository and install Olive

Refer to the instructions in the [examples README](../README.md) to clone the repository and install Olive.


We use the same olive workflow config files and scripts as the DirectML examples. The only difference is the `--provider cuda` option provided to the `stable_diffusion.py`  and `stable_diffusion_xl.py` scripts.

So, cd into the corresponding DirectML example folder from the root of the cloned repository:

**_Stable Diffusion_**
```bash
cd examples/directml/stable_diffusion
```

**_Stable Diffusion XL_**
```bash
cd examples/directml/stable_diffusion_xl
```

#### Install onnxruntime
This example requires the latest onnxruntime-gpu code which can either be built from source or installed from the nightly builds. The following command can be used to install the latest nightly build of onnxruntime-gpu:

```bash
# uninstall any pre-existing onnxruntime packages
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml ort-nightly ort-nightly-gpu ort-nightly-directml

# install onnxruntime-gpu nightly build
pip install ort-nightly-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
```

#### Install other dependencies
Install the necessary python packages:

```bash
python -m pip install -r requirements-common.txt
```

### Conversion to ONNX and Latency Optimization

The easiest way to optimize the pipeline is with the `stable_diffusion.py` and `stable_diffusion_xl.py` scripts. These scripts will enumerate the `config_<model_name>.json` files and optimize each with Olive, then gather the optimized models into a directory structure suitable for testing inference.

**_Stable Diffusion_**
```bash
# default model_id is "runwayml/stable-diffusion-v1-5"
python stable_diffusion.py --provider cuda --optimize
```

**_Stable Diffusion XL_**
```bash
# default model_id is "stabilityai/stable-diffusion-xl-base-1.0"
python stable_diffusion_xl.py --provider cuda --optimize

# or specify a different model_id
python stable_diffusion_xl.py --provider cuda --model_id stabilityai/stable-diffusion-xl-refiner-1.0 --optimize
```

Once the script successfully completes:
- The optimized ONNX pipeline will be stored under `models/optimized-cuda/[model_id]` (for example `models/optimized-cuda/runwayml/stable-diffusion-v1-5` or `models/optimized-cuda/stabilityai/stable-diffusion-xl-base-1.0`).
- The unoptimized ONNX pipeline (models converted to ONNX, but not run through transformer optimization pass) will be stored under `models/unoptimized/[model_id]` (for example `models/unoptimized/runwayml/stable-diffusion-v1-5` or `models/unoptimized/stabilityai/stable-diffusion-xl-base-1.0`).

Re-running the script with `--optimize` will delete the output models, but it will *not* delete the Olive cache. Subsequent runs will complete much faster since it will simply be copying previously optimized models; you may use the `--clean_cache` option to start from scratch (not typically used unless you are modifying the scripts, for example).

## Test Inference

Test ONNX runtime inference with the optimized models using `OnnxStableDiffusionPipeline`:

**_Stable Diffusion_**
```bash
python stable_diffusion.py --provider cuda --num_images 2
```
Inference will loop until the generated image passes the safety checker (otherwise you would see black images). The result will be saved as `result_<i>.png` on disk.


**_Stable Diffusion XL_**
```bash
python stable_diffusion_xl.py --provider cuda --num_images 2
```
The result will be saved as `result_<i>.png` on disk.


Refer to the corresponding section in the DirectML READMEs for more details on the test inference options:
- [Stable Diffusion](../directml/stable_diffusion/README.md#test-inference)
- [Stable Diffusion XL](../directml/stable_diffusion_xl/README.md#test-inference)
