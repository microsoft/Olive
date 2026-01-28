# Fine-Tuning Diffusion Models with Olive

*Author: Xiaoyu Zhang*
*Created: 2026-01-26*

This guide shows you how to fine-tune Stable Diffusion and Flux models with LoRA adapters using Olive. You can use either:

- **CLI**: Quick start with `olive diffusion-lora` command
- **JSON Configuration**: Full control over data preprocessing and training options

## Overview

Olive provides a simple CLI command to train LoRA (Low-Rank Adaptation) adapters for diffusion models. This allows you to:

- Teach your model new artistic styles
- Train it to generate specific subjects (DreamBooth)
- Customize image generation without modifying the full model weights

### Supported Models

| Model Type | Example Models | Default Resolution |
|------------|----------------|-------------------|
| SD 1.5 | `runwayml/stable-diffusion-v1-5` | 512x512 |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | 1024x1024 |
| Flux | `black-forest-labs/FLUX.1-dev` | 1024x1024 |

## Quick Start

### Basic LoRA Training

Train a LoRA adapter on your own images:

```bash
# Using a local image folder
olive diffusion-lora \
    -m runwayml/stable-diffusion-v1-5 \
    -d /path/to/your/images \
    -o my-style-lora

# Using a HuggingFace dataset
olive diffusion-lora \
    -m runwayml/stable-diffusion-v1-5 \
    --data_name linoyts/Tuxemon \
    --caption_column prompt \
    -o tuxemon-lora
```

### DreamBooth Training

Train the model to generate a specific subject (person, pet, object):

```bash
olive diffusion-lora \
    -m stabilityai/stable-diffusion-xl-base-1.0 \
    --model_variant sdxl \
    -d /path/to/subject/images \
    --dreambooth \
    --instance_prompt "a photo of sks dog" \
    --with_prior_preservation \
    --class_prompt "a photo of a dog" \
    -o my-dog-lora
```

## Data Sources

Olive supports two ways to provide training data:

### 1. Local Image Folder

Organize your images in a folder with optional caption files:

```
my_training_data/
├── image1.jpg
├── image1.txt     # Caption: "a beautiful sunset over mountains"
├── image2.png
├── image2.txt     # Caption: "a cat sitting on a couch"
└── subfolder/
    ├── image3.jpg
    └── image3.txt
```

Each `.txt` file contains the caption/prompt for the corresponding image.

**No captions?** No problem! Use the `auto_caption` preprocessing step to automatically generate captions using BLIP-2 or Florence-2 models. See the [Data Preprocessing](#data-preprocessing) section for details.

### 2. HuggingFace Dataset

Use any image dataset from the HuggingFace Hub. Specify `--data_name` with optional `--image_column` and `--caption_column` parameters.

## Command Reference

For the complete list of CLI options, see the [Diffusion LoRA CLI Reference](https://microsoft.github.io/Olive/reference/cli.html#diffusion-lora).

```bash
olive diffusion-lora --help
```

## Using the Trained LoRA

After training, load your LoRA adapter with diffusers:

```python
from diffusers import DiffusionPipeline
import torch

# Load base model (works for SD, SDXL, Flux)
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA adapter
pipe.load_lora_weights("./my-lora-output/adapter")

# Generate images
image = pipe("a beautiful landscape").images[0]
image.save("output.png")
```

## Tips and Best Practices

### Dataset Preparation

1. **Image Quality**: Use high-quality, consistent images. Aim for 10-50 images for style transfer, 5-20 for DreamBooth.

2. **Captions**: Write descriptive captions that include the key elements you want the model to learn. For DreamBooth, use a unique trigger word (e.g., "sks") that doesn't conflict with existing concepts.

3. **Resolution**: Images don't need to match the training resolution exactly. Olive automatically handles aspect ratio bucketing and resizing, but remember to set `--model_variant sdxl/flux` or `--base_resolution 1024` when training SDXL/Flux so preprocessing runs at the correct size.

### Training Parameters

1. **LoRA Rank (`-r`)**:
   - SD 1.5/SDXL: 4-16 is usually sufficient
   - Flux: Use 16-64 for better quality

2. **Training Steps**:
   - Style transfer: 1000-3000 steps
   - DreamBooth: 500-1500 steps

3. **Learning Rate**:
   - Start with `1e-4` and adjust based on results
   - Lower (e.g., `5e-5`) if overfitting, higher (e.g., `2e-4`) if underfitting

4. **Prior Preservation**: Always use `--with_prior_preservation` for DreamBooth to prevent the model from forgetting general concepts.

### Hardware Requirements (guidelines)

| Model | Minimum VRAM | Recommended VRAM |
|-------|--------------|------------------|
| SD 1.5 | 8 GB | 12+ GB |
| SDXL | 16 GB | 24+ GB |
| Flux | 24 GB | 40+ GB |


## Advanced: Custom Configuration

For more control, you can use Olive's configuration file instead of CLI options:

```json
{
    "input_model": {
        "type": "DiffusersModel",
        "model_path": "stabilityai/stable-diffusion-xl-base-1.0"
    },
    "data_configs": [{
        "name": "train_data",
        "type": "ImageDataContainer",
        "load_dataset_config": {
            "type": "huggingface_dataset",
            "params": {
                "data_name": "linoyts/Tuxemon",
                "split": "train",
                "image_column": "image",
                "caption_column": "prompt"
            }
        },
        "pre_process_data_config": {
            "type": "image_lora_preprocess",
            "params": {
                "base_resolution": 1024,
                "steps": {
                    "auto_caption": {"model_type": "florence2"},
                    "aspect_ratio_bucketing": {}
                }
            }
        }
    }],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "r": 16,
            "alpha": 16,
            "training_args": {
                "max_train_steps": 2000,
                "learning_rate": 1e-4,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "mixed_precision": "bf16"
            }
        }
    },
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "gpu"}]
        }
    },
    "host": "local_system",
    "target": "local_system",
    "output_dir": "my-lora-output"
}
```

Run with:

```bash
olive run --config my_lora_config.json
```

## Data Preprocessing

Olive supports automatic data preprocessing including image filtering, auto-captioning, tagging, and aspect ratio bucketing.

**CLI** only supports basic aspect ratio bucketing via `--base_resolution`. For advanced preprocessing (auto-captioning, filtering, tagging), use a JSON configuration file.

For detailed preprocessing options and examples, see the [SD LoRA Feature Documentation](https://microsoft.github.io/Olive/features/sd-lora.html).

## Export to ONNX and Run Inference

After fine-tuning, you can merge the LoRA adapter into the base model and export the pipeline to ONNX with Olive's CLI, then run inference using ONNX Runtime.

### 1. Export with the CLI

Use `capture-onnx-graph` to export the base components together with your LoRA adapter:

```bash
olive capture-onnx-graph \
    -m stabilityai/stable-diffusion-xl-base-1.0 \
    -a my-lora-output/adapter \
    --output_path sdxl-lora-onnx
```

### Multi LoRA + inference

Want to combine multiple adapters or see a full inference notebook? Check [sd_multilora.ipynb](https://github.com/microsoft/Olive/blob/main/notebooks/sd_multilora/sd_multilora.ipynb) for an end-to-end example covering multi-LoRA composition and ONNX Runtime inference.

## Related Resources

- [DreamBooth Paper](https://arxiv.org/abs/2208.12242)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
