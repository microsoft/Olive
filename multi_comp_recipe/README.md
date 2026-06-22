# Multi-Component Model Optimization Recipes

These recipes demonstrate **Flow A — export first, then per-component optimization**: export a
multi-component model to ONNX once, then run a single Olive config whose `builds` apply a
**different pipeline to each component**.

The flow is two explicit steps:

1. **Export** the model to a directory of per-component ONNX subfolders using the Olive CLI with the
   Mobius builder.
2. **Optimize** by pointing an Olive config at that directory; each component subfolder becomes a
   selectable component that a `build` can target.

There is no need to memorize component names: each exported component lives in its own folder, and
Olive loads the export directory as a `CompositeModel` whose **component names are the subfolder
names**.

---

## Prerequisites

```
pip install olive-ai
pip install mobius-ai
```

Exporting a diffusion pipeline also needs `diffusers`/`transformers` and access to the model on
Hugging Face (Stable Diffusion 3 is a gated model — accept its license and `huggingface-cli login`
first).

---

## Recipe 1 — Stable Diffusion 3 (`sd3_optimize_components.json`)

### Step 1 — Export with the CLI

```
olive capture-onnx-graph --model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers --use_mobius_builder --output_path exported_pkg
```

Mobius exports each neural-network component to its own subfolder:

```
exported_pkg/
  transformer/model.onnx     # MMDiT denoising backbone
  vae_encoder/model.onnx
  vae_decoder/model.onnx
```

> **Note.** Mobius's diffusers builder exports the **transformer** backbone and the **VAE**
> (encoder + decoder). The CLIP/T5 **text encoders are not exported by Mobius** and are left to the
> original pipeline. The exact subfolders depend on the pipeline; the optimize config below only
> needs `builds` for the components you actually want to optimize.

### Step 2 — Optimize each component

Run from the directory that contains `exported_pkg/`:

```
olive run --config sd3_optimize_components.json
```

This applies a different pipeline per component:

| component     | pipeline           | intent                                   |
|---------------|--------------------|------------------------------------------|
| `transformer` | `dynamic_quant`    | INT8-quantize the heavy denoising backbone |
| `vae_encoder` | `to_fp16`          | keep the VAE in FP16 to preserve quality |
| `vae_decoder` | `to_fp16`          | keep the VAE in FP16 to preserve quality |

Output:

```
out/transformer/    # INT8 transformer
out/vae_encoder/    # FP16 VAE encoder
out/vae_decoder/    # FP16 VAE decoder
```

Each build writes one optimized component; components without a build stay as exported.

---

## Recipe 2 — Vision-Language Model (`vlm_optimize_components.json`)

Same two-step Flow A for a VLM, using `Qwen/Qwen3-VL-2B-Instruct`.

### Step 1 — Export

```
olive capture-onnx-graph --model_name_or_path Qwen/Qwen3-VL-2B-Instruct --use_mobius_builder --output_path exported_vlm_pkg
```

Mobius exports this model as three components, each in its own subfolder:

```
exported_vlm_pkg/
  decoder/model.onnx
  vision_encoder/model.onnx
  embedding/model.onnx
```

### Step 2 — Optimize

```
olive run --config vlm_optimize_components.json
```

| component        | pipeline        | intent                              |
|------------------|-----------------|-------------------------------------|
| `decoder`        | `dynamic_quant` | INT8-quantize the language decoder  |
| `vision_encoder` | `to_fp16`       | keep the vision tower in FP16       |
| `embedding`      | `to_fp16`       | keep the embedding in FP16          |

> The three component names (`decoder`, `vision_encoder`, `embedding`) are exactly what Mobius
> produces for `Qwen/Qwen3-VL-2B-Instruct`. For a different VLM, adjust the component names in the
> config to match the subfolder names your export actually produced.

---

## Notes

- The passes here (`OnnxFloatToFloat16`, `OnnxDynamicQuantization`) are **illustrative** and chosen
  to run without calibration data. Swap in `OrtTransformersOptimization`, `OnnxStaticQuantization`
  (with a `data_config`), or other ONNX passes for production-quality optimization.
- The recipes target the **CPU** EP so they run anywhere. For GPU deployment, change the
  `execution_providers` to e.g. `["CUDAExecutionProvider"]` and the device to `"gpu"`.
- `builds.components` selects which exported components to optimize. Only the components with a build
  are touched; the rest remain as exported.
