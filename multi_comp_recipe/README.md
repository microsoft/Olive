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
  text_encoder/model.onnx    # CLIP-L text encoder
  text_encoder_2/model.onnx  # CLIP-G text encoder
  text_encoder_3/model.onnx  # T5-XXL text encoder
  transformer/model.onnx     # MMDiT denoising backbone
  vae_encoder/model.onnx
  vae_decoder/model.onnx
```

> **Note.** The exact subfolders depend on the pipeline; the optimize config below only
> needs `builds` for the components you actually want to optimize.

### Step 2 — Optimize each component

Run from the directory that contains `exported_pkg/`:

```
olive run --config sd3_optimize_components.json
```

This applies a different pipeline per component:

| component        | pipeline           | intent                                   |
|------------------|--------------------|------------------------------------------|
| `transformer`    | `dynamic_quant`    | INT8-quantize the heavy denoising backbone |
| `text_encoder_3` | `to_fp16`          | keep T5-XXL in FP16                     |
| `vae_encoder`    | `to_fp16`          | keep the VAE in FP16 to preserve quality |
| `vae_decoder`    | `to_fp16`          | keep the VAE in FP16 to preserve quality |

Output:

```
out/transformer/    # INT8 transformer
out/text_encoder_3/ # FP16 T5-XXL
out/vae_encoder/    # FP16 VAE encoder
out/vae_decoder/    # FP16 VAE decoder
```

Each build writes one optimized component; components without a build stay as exported.

### Step 3 — Inference

Run end-to-end image generation with the exported ONNX models:

```
python sd3_inference.py --prompt "A photo of a cat sitting on a windowsill" --steps 28 --output result.png
```

The inference script (`sd3_inference.py`) uses:
- **Text encoding**: ONNX Runtime with exported CLIP-L, CLIP-G, and T5-XXL encoders (run once)
- **Denoising**: ONNX Runtime with the exported SD3 transformer (28 steps)
- **VAE decoding**: ONNX Runtime with the exported VAE decoder

Options:
```
--prompt TEXT       Text prompt for image generation
--steps N           Number of denoising steps (default: 28)
--seed N            Random seed (default: 42)
--output PATH       Output image path (default: sd3_output.png)
--onnx_dir DIR      Path to exported model directory (default: exported_sd3_full2)
```

> **Note.** SD3 is a gated model — you need `huggingface-cli login` or set `HF_TOKEN` to export.
> The tokenizers (CLIP and T5) still run via the `transformers` library.

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

### Step 3 — Inference with ORT GenAI

Run text generation with the exported ONNX models using **onnxruntime-genai**:

```bash
# Text-only
python vlm_inference.py --prompt "The capital of France is"

# With image input
python vlm_inference.py --prompt "Describe this image." --image photo.jpg

# Custom settings
python vlm_inference.py --model_dir exported_vlm_pkg --max_new_tokens 256
```

The inference script (`vlm_inference.py`) uses ORT GenAI which handles:
- **Tokenization**: Built-in tokenizer from saved HF tokenizer files
- **Embedding**: ONNX `embedding/model.onnx` (token embed + image/audio feature mixing)
- **Vision encoding**: ONNX `vision_encoder/model.onnx` (when `--image` is provided)
- **Decoding**: ONNX `decoder/model.onnx` with KV cache (autoregressive generation)

Options:
```
--prompt TEXT           Text prompt
--image PATH            Optional image file for multimodal input
--max_new_tokens N      Maximum tokens to generate (default: 128)
--model_dir DIR         Path to exported model directory (default: exported_vlm_pkg)
```

#### Setup requirements

The export directory needs these files alongside the ONNX models:

```
exported_vlm_pkg/
  genai_config.json          # Model type, I/O mappings, search config
  tokenizer.json             # HF tokenizer
  tokenizer_config.json
  vision_processor.json      # Vision preprocessing config
  audio_processor.json       # Audio preprocessing config (for Phi-4-multimodal)
  decoder/model.onnx
  vision_encoder/model.onnx
  embedding/model.onnx
  audio_encoder/model.onnx   # Optional (Phi-4-multimodal)
```

To create `genai_config.json` and tokenizer files after export:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
tokenizer.save_pretrained("exported_vlm_pkg")
```

For the `genai_config.json` structure, see the
[Mobius phi4mm example](https://github.com/microsoft/mobius/blob/main/examples/phi4mm_ort_genai.py)
which writes the config automatically.

> **Note.** Install `onnxruntime-genai` (`pip install onnxruntime-genai`) to use this script.

---

## Notes

- The passes here (`OnnxFloatToFloat16`, `OnnxDynamicQuantization`) are **illustrative** and chosen
  to run without calibration data. Swap in `OrtTransformersOptimization`, `OnnxStaticQuantization`
  (with a `data_config`), or other ONNX passes for production-quality optimization.
- The recipes target the **CPU** EP so they run anywhere. For GPU deployment, change the
  `execution_providers` to e.g. `["CUDAExecutionProvider"]` and the device to `"gpu"`.
- `builds.components` selects which exported components to optimize. Only the components with a build
  are touched; the rest remain as exported.
