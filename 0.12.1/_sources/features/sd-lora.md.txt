# Diffusion Model LoRA Training

Olive provides the `SDLoRA` pass for training LoRA (Low-Rank Adaptation) adapters on diffusion models. This enables efficient fine-tuning of large image generation models with minimal GPU memory requirements.

## Supported Models

| Model Type | Examples | Resolution | Notes |
|------------|----------|------------|-------|
| **SD 1.5** | `runwayml/stable-diffusion-v1-5` | 512 | Standard Stable Diffusion |
| **SDXL** | `stabilityai/stable-diffusion-xl-base-1.0` | 1024 | Dual CLIP encoders |
| **Flux** | `black-forest-labs/FLUX.1-dev` | 1024 | DiT architecture, requires bfloat16 |

## Quick Start with CLI

The easiest way to train a LoRA adapter is using the `olive diffusion-lora` command.

### Basic Usage

```bash
# Train with local images
olive diffusion-lora -m runwayml/stable-diffusion-v1-5 -d ./train_images

# Train with HuggingFace dataset
olive diffusion-lora -m runwayml/stable-diffusion-v1-5 --data_name linoyts/Tuxemon --caption_column prompt

# Train SDXL
olive diffusion-lora -m stabilityai/stable-diffusion-xl-base-1.0 -d ./train_images

# Train Flux
olive diffusion-lora -m black-forest-labs/FLUX.1-dev -d ./train_images -r 32
```

### CLI Options

#### Model Options

| Option | Description |
|--------|-------------|
| `-m, --model_name_or_path` | HuggingFace model name or local path (required) |
| `-o, --output_path` | Output path for LoRA adapter (default: `diffusion-lora-adapter`) |
| `--model_variant` | Model variant: `auto`, `sd15`, `sdxl`, `flux` (default: `auto`) |

#### LoRA Options

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --lora_r` | 16 | LoRA rank (SD: 4-16, Flux: 16-64) |
| `--alpha` | Same as r | LoRA alpha for scaling |
| `--lora_dropout` | 0.0 | LoRA dropout probability |
| `--target_modules` | Auto | Target modules (comma-separated) |
| `--merge_lora` | False | Merge LoRA into base model |

#### Data Options

| Option | Description |
|--------|-------------|
| `-d, --data_dir` | Path to local image folder |
| `--data_name` | HuggingFace dataset name |
| `--data_split` | Dataset split (default: `train`) |
| `--image_column` | Image column name (default: `image`) |
| `--caption_column` | Caption column name |
| `--base_resolution` | Base resolution (auto-detected from model type) |

#### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max_train_steps` | 1000 | Maximum training steps |
| `--learning_rate` | 1e-4 | Learning rate |
| `--train_batch_size` | 1 | Training batch size |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `--mixed_precision` | bf16 | Mixed precision: `no`, `fp16`, `bf16` |
| `--lr_scheduler` | constant | LR scheduler type |
| `--lr_warmup_steps` | 0 | Warmup steps |
| `--seed` | None | Random seed |

#### DreamBooth Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dreambooth` | False | Enable DreamBooth training |
| `--prior_loss_weight` | 1.0 | Prior preservation loss weight |

#### Flux Options

| Option | Default | Description |
|--------|---------|-------------|
| `--guidance_scale` | 3.5 | Guidance scale for Flux training |

### CLI Examples

```bash
# SD 1.5 with custom training settings
olive diffusion-lora \
    -m runwayml/stable-diffusion-v1-5 \
    -d ./train_images \
    -r 4 \
    --max_train_steps 500 \
    --learning_rate 5e-5 \
    -o my-lora

# SDXL with HuggingFace dataset
olive diffusion-lora \
    -m stabilityai/stable-diffusion-xl-base-1.0 \
    --data_name linoyts/Tuxemon \
    --caption_column prompt \
    -r 16 \
    --max_train_steps 2000

# Flux with higher rank
olive diffusion-lora \
    -m black-forest-labs/FLUX.1-dev \
    -d ./train_images \
    -r 32 \
    --mixed_precision bf16 \
    --guidance_scale 3.5

# DreamBooth training
olive diffusion-lora \
    -m runwayml/stable-diffusion-v1-5 \
    -d ./train_images \
    --dreambooth \
    --prior_loss_weight 1.0

# Merge LoRA into base model
olive diffusion-lora \
    -m runwayml/stable-diffusion-v1-5 \
    -d ./train_images \
    --merge_lora
```

## Training Data Structure

Prepare your training images with corresponding caption files:

```
train_images/
├── image1.png
├── image1.txt    # Contains: "a photo of sks dog"
├── image2.jpg
├── image2.txt    # Contains: "sks dog playing in the park"
└── ...
```

## Configuration File

For more complex workflows or integration with other Olive passes, use a JSON configuration file.

### Minimal Configuration

```json
{
    "input_model": {
        "type": "DiffusersModel",
        "model_path": "runwayml/stable-diffusion-v1-5"
    },
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "gpu"}]
        }
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "ImageDataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            }
        }
    ],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data"
        }
    },
    "host": "local_system",
    "target": "local_system",
    "output_dir": "output"
}
```

Run with:
```bash
olive run --config config.json
```

### Using HuggingFace Datasets

```json
{
    "input_model": {
        "type": "DiffusersModel",
        "model_path": "runwayml/stable-diffusion-v1-5"
    },
    "data_configs": [
        {
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
            }
        }
    ],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "r": 4,
            "training_args": {
                "max_train_steps": 1000,
                "train_batch_size": 4
            }
        }
    }
}
```

## SDLoRA Pass Configuration

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_variant` | str | `"auto"` | Model variant: `"sd15"`, `"sdxl"`, `"flux"`, or `"auto"` |
| `r` | int | 16 | LoRA rank |
| `alpha` | float | None | LoRA alpha (defaults to r) |
| `lora_dropout` | float | 0.0 | Dropout probability |
| `target_modules` | list | None | Target modules (auto-detected if None) |
| `merge_lora` | bool | False | Merge LoRA into base model |

### DreamBooth

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dreambooth` | bool | False | Enable DreamBooth training (for learning specific subjects) |
| `prior_loss_weight` | float | 1.0 | Weight of prior preservation loss (only when dreambooth=True) |

### Training Arguments

Configure via `training_args`:

```json
{
    "type": "SDLoRA",
    "train_data_config": "train_data",
    "training_args": {
        "learning_rate": 1e-4,
        "max_train_steps": 1000,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": true,
        "mixed_precision": "bf16",
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "checkpointing_steps": 500,
        "logging_steps": 10
    }
}
```

| Argument | Default | Description |
|----------|---------|-------------|
| `learning_rate` | 1e-4 | Learning rate |
| `max_train_steps` | 1000 | Maximum training steps |
| `train_batch_size` | 1 | Batch size |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `gradient_checkpointing` | True | Enable gradient checkpointing |
| `mixed_precision` | `"bf16"` | Mixed precision mode (`"fp16"`, `"bf16"`, `"no"`) |
| `lr_scheduler` | `"constant"` | LR scheduler type |
| `lr_warmup_steps` | 0 | Warmup steps |
| `max_grad_norm` | 1.0 | Max gradient norm |
| `snr_gamma` | None | SNR gamma for Min-SNR weighting |
| `checkpointing_steps` | 500 | Save checkpoint every N steps |
| `logging_steps` | 10 | Log every N steps |
| `seed` | None | Random seed |
| `guidance_scale` | 3.5 | Flux only: guidance scale |
| `use_prodigy` | False | Flux only: use Prodigy optimizer |
| `prodigy_beta3` | None | Flux only: Prodigy beta3 parameter |

## Data Configuration

Use `ImageDataContainer` with `image_lora_preprocess` for automatic data preprocessing.

### Local Image Folder

```json
{
    "data_configs": [
        {
            "name": "train_data",
            "type": "ImageDataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            }
        }
    ]
}
```

### HuggingFace Dataset

```json
{
    "data_configs": [
        {
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
            }
        }
    ]
}
```

### Preprocessing Chain

The preprocessing chain supports multiple steps:

| Step | Default | Description |
|------|---------|-------------|
| `image_filtering` | Disabled | Filter low quality images |
| `auto_caption` | Disabled | Generate captions with VLM |
| `auto_tagging` | Disabled | Generate tags with WD14 |
| `image_resizing` | Disabled | Resize images to fixed size |
| `aspect_ratio_bucketing` | Enabled | Group by aspect ratio |

Default preprocessing is `aspect_ratio_bucketing` with `base_resolution=512`.

### Custom Preprocessing

```json
{
    "data_configs": [
        {
            "name": "train_data",
            "type": "ImageDataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            },
            "pre_process_data_config": {
                "type": "image_lora_preprocess",
                "params": {
                    "base_resolution": 1024,
                    "output_dir": "resized_images",
                    "steps": {
                        "auto_caption": {
                            "model_type": "florence2",
                            "trigger_word": "sks"
                        },
                        "aspect_ratio_bucketing": {}
                    }
                }
            }
        }
    ]
}
```

### Auto Captioning

Automatically generate captions using vision-language models:

#### Supported Captioning Models

| Model | Default Model Name | Features |
|-------|-------------------|----------|
| `blip2` | `Salesforce/blip2-opt-2.7b` | General captions |
| `florence2` | `microsoft/Florence-2-large` | Detailed descriptions |

#### Auto Caption Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `"blip2"` | Captioning model (`"blip2"` or `"florence2"`) |
| `model_name` | None | Custom model path |
| `trigger_word` | None | Trigger word to prepend to all captions (e.g., `"sks"`) |
| `overwrite` | False | Overwrite existing captions |
| `device` | `"cuda"` | Device for inference |

### Aspect Ratio Bucketing

Groups images by aspect ratio to minimize padding and improve training quality.

Set `base_resolution` based on your model:
- SD 1.5: `base_resolution=512` (default)
- SDXL/Flux: `base_resolution=1024`

## Model-Specific Examples

### Stable Diffusion 1.5

```json
{
    "input_model": {
        "type": "DiffusersModel",
        "model_path": "runwayml/stable-diffusion-v1-5"
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "ImageDataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            }
        }
    ],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "r": 4
        }
    }
}
```

### SDXL

```json
{
    "input_model": {
        "type": "DiffusersModel",
        "model_path": "stabilityai/stable-diffusion-xl-base-1.0"
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "ImageDataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            },
            "pre_process_data_config": {
                "type": "image_lora_preprocess",
                "params": {
                    "base_resolution": 1024
                }
            }
        }
    ],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "r": 16
        }
    }
}
```

### Flux

```json
{
    "input_model": {
        "type": "DiffusersModel",
        "model_path": "black-forest-labs/FLUX.1-dev"
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "ImageDataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            },
            "pre_process_data_config": {
                "type": "image_lora_preprocess",
                "params": {
                    "base_resolution": 1024
                }
            }
        }
    ],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "r": 32,
            "training_args": {
                "mixed_precision": "bf16",
                "guidance_scale": 3.5
            }
        }
    }
}
```

**Note:** Flux requires `bfloat16` - the pass will automatically switch from `float16` if needed.

## Inference

After training, load the LoRA weights using diffusers:

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA
pipe.load_lora_weights("output/adapter")
pipe.fuse_lora(lora_scale=1.0)

# Generate
image = pipe(
    "a photo of sks dog in a garden",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("output.png")
```

## Tips

1. **Memory**: Enable `gradient_checkpointing` and reduce `train_batch_size` if OOM
2. **Quality**: Use 10-20 high-quality, diverse training images
3. **Captions**: Include a unique trigger word (e.g., "sks") in all captions
4. **LoRA Rank**: Start with r=4-16 for SD, r=16-64 for Flux
5. **Overfitting**: Monitor training loss; reduce steps if outputs look too similar to training data
6. **Inference Scale**: Use `lora_scale=0.7-0.8` if LoRA effect is too strong

## Dependencies

Install required dependencies:

```bash
pip install olive-ai[sd-lora]

# Or manually:
pip install accelerate>=0.30.0 peft diffusers>=0.25.0 transformers>=4.30.0
```

For auto-captioning:
```bash
pip install transformers>=4.30.0  # For BLIP-2 and Florence-2
```
