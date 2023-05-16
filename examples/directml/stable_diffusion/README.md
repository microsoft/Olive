# Stable Diffusion Optimization with DirectML <!-- omit in toc -->

    ⚠️ THIS SAMPLE IS A WORK IN PROGRESS AND REQUIRES ONNXRUNTIME-DIRECTML 1.15+ (NOT YET RELEASED) ⚠️

This sample shows how to optimize [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) to run with ONNX Runtime and DirectML.

Stable Diffusion comprises multiple PyTorch models tied together into a *pipeline*. This Olive sample will convert each PyTorch model to ONNX, and then run the converted ONNX models through the `OrtTransformersOptimization` pass. The transformer optimization pass performs several time-consuming graph transformations that make the models more efficient for inference at runtime. Output models are only guaranteed to be compatible with onnxruntime-directml (1.15+).

Contents:
- [Setup](#setup)
- [Conversion to ONNX and Latency Optimization](#conversion-to-onnx-and-latency-optimization)
- [Test Inference](#test-inference)
- [LoRA Models (Experimental)](#lora-models-experimental)
- [Issues](#issues)
- [Stable Diffusion Pipeline](#stable-diffusion-pipeline)

# Setup

Make sure that your Python environment has `onnxruntime-directml` along with other dependencies in this sample's [requirements.txt](requirements.txt):

```
pip install -r requirements.txt
```

# Conversion to ONNX and Latency Optimization

The easiest way to optimize the pipeline is with the `stable_diffusion.py` helper script:

```
python stable_diffusion.py --optimize
```

The stable diffusion models are large, and the optimization process is resource intensive. It is recommended to run optimization on a system with a minimum of 16GB of memory (preferably 32GB). Expect optimization to take several minutes (especially the U-Net model).

Once the script successfully completes:
- The optimized ONNX pipeline will be stored under `models/optimized/runwayml/stable-diffusion-v1-5`.
- The unoptimized ONNX pipeline (models converted to ONNX, but not run through transformer optimization pass) will be stored under `models/unoptimized/runwayml/stable-diffusion-v1-5`.

Re-running the script with `--optimize` will delete the output models, but it will *not* delete the Olive cache. Subsequent runs will complete much faster since it will simply be copying previously optimized models; you may use the `--clean_cache` option to start from scratch (not typically used unless you are modifying the scripts, for example).

# Test Inference

This sample code is primarily intended to illustrate model optimization with Olive, but it also provides a simple interface for testing inference with the ONNX models. Inference is done by creating an `OnnxStableDiffusionPipeline` from the saved models, which leans on ONNX runtime for inference of the core models (text encoder, u-net, decoder, and safety checker).

Invoke the script with `--interactive` (and optionally `--num_images <count>`) to present a simple GUI where you may enter a prompt and generate images.

```
python stable_diffusion.py --interactive --num_images 2
Loading models into ORT session...

Inference Batch Start (batch size = 1).
100%|███████████████████████████████████████████████| 51/51 [00:02<00:00, 23.93it/s]
Generated result_0.png
Inference Batch End (1/1 images passed the safety checker).

Inference Batch Start (batch size = 1).
100%|███████████████████████████████████████████████| 51/51 [00:01<00:00, 26.22it/s]
Generated result_1.png
Inference Batch End (1/1 images passed the safety checker).
```

Inference will loop until the generated image passes the safety checker (otherwise you would see black images). The result will be saved as `result_0.png` on disk, which is then loaded and displayed in the UI. Example below:

![example output](readme/example.png)

Run `python stable_diffusion.py --help` for additional options. A few particularly relevant ones:
- `--model_id <string>` : name of a stable diffusion model ID hosted by huggingface.co. This script has been tested with the following:
  - `CompVis/stable-diffusion-v1-4`
  - `runwayml/stable-diffusion-v1-5` (default)
  - LoRA variants of the above base models may work as well. See [LoRA Models (Experimental)](#lora-models-experimental).
- `--num_inference_steps <count>` : the default value is 50, but a lower value may produce sufficiently high quality images while taking less time overall.

# LoRA Models (Experimental)

This script has limited support for optimizing [LoRA variants of a base Stable Diffusion model](https://huggingface.co/docs/diffusers/main/en/training/lora). When optimizing or running inference, specify the LoRA model ID instead of the base model ID:

```
# Optimization:
python .\stable_diffusion.py --optimize --model_id "sayakpaul/sd-model-finetuned-lora-t4"

# Inference test:
python .\stable_diffusion.py --interactive --model_id "sayakpaul/sd-model-finetuned-lora-t4"
```

In the above example, `sayakpaul/sd-model-finetuned-lora-t4` is based on `CompVis/stable-diffusion-v1-4`, so the text encoder, VAE decoder, and safety checker models will be optimized just as if you were optimizing `CompVis/stable-diffusion-v1-4`. The U-Net model, however, will have the custom LoRA weights applied with a default scale of 1.0 (if you wish to change the scale, modify its default value in `user_script.py:merge_lora_weights`).

This script does not yet support loading [LoRA weights from a .safetensors file](https://github.com/huggingface/diffusers/issues/3064). For now, you can only use a model ID of a pretrained model hosted inside a model repo on huggingface.co (e.g. see https://huggingface.co/lora-library).

**Implementation Details**

LoRA adds additional linear layers (attention processors) to the base PyTorch model. The added layers have their own small set of weights ("LoRA weights"), which allows users to replace only a portion of the the full model weights when switching between LoRA variants. Without preprocessing, the additional LoRA layers reduce inference speed since there are more layers in the network architecture. In practice, the added LoRA layers can be folded into existing layers of the base model to completely remove the inference overhead; however, once the LoRA weights are merged with the base weights it is no longer possible to swap out new LoRA weights (the model is "baked").

Olive merges the LoRA weights into the base model before conversion to ONNX (see `user_script.py:merge_lora_weights`), which means the output models are fully baked. This approach simplifies downstream ONNX-based graph optimizations and enables performance-critical fusions (e.g. multi-head attention) that will not occur if the injected LoRA layers remain in the graph. As a consequence, if you want to switch LoRA weights you must reoptimize the affected models (generally U-Net). Retaining the flexibility of pluggable LoRA weights in the output ONNX models would require the LoRA weights be saved as external data along with fusion of the attention processors at runtime.

# Issues

If you run into the following error while optimizing models, it is likely that your local HuggingFace cache has an incomplete copy of the stable diffusion model pipeline. Deleting `C:\users\<username>\.cache\huggingface` should resolve the issue by ensuring a fresh copy is downloaded.

```
OSError: Can't load tokenizer for 'C:\Users\<username>\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\<sha>'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'C:\Users\<username>\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\aa9ba505e1973ae5cd05f5aedd345178f52f8e6a' is the correct path to a directory containing all relevant files for a CLIPTokenizer tokenizer.
```

# Stable Diffusion Pipeline

The figure belows a high-level overview of the Stable Diffusion pipeline, and is based on a figure from [Hugging Face Blog](https://huggingface.co/blog/stable_diffusion) that covers Stable Diffusion with Diffusers library. The blue boxes are the converted & optimized ONNX models. The gray boxes remain implemented by diffusers library when using this example for inference; a custom pipeline may implement the full pipeline without leveraging Python or the diffusers library.

![sd pipeline](readme/pipeline.png)