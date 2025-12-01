# Diffusion Model LoRA Training

Olive provides the `SDLoRA` pass for training LoRA (Low-Rank Adaptation) adapters on diffusion models. This enables efficient fine-tuning of large image generation models with minimal GPU memory requirements.

## Supported Models

| Model Type | Examples | Resolution | Notes |
|------------|----------|------------|-------|
| **SD 1.5** | `runwayml/stable-diffusion-v1-5` | 512 | Standard Stable Diffusion |
| **SDXL** | `stabilityai/stable-diffusion-xl-base-1.0` | 1024 | Dual CLIP encoders |
| **Flux** | `black-forest-labs/FLUX.1-dev` | 1024 | DiT architecture, requires bfloat16 |

## Quick Start

### Minimal Configuration

```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "stabilityai/stable-diffusion-xl-base-1.0"
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
            "type": "SDLoRADataContainer",
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

### Training Data Structure

Prepare your training images with corresponding caption files:

```
train_images/
├── image1.png
├── image1.txt    # Contains: "a photo of sks dog"
├── image2.jpg
├── image2.txt    # Contains: "sks dog playing in the park"
└── ...
```

## SDLoRA Pass Configuration

### Basic Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | `"auto"` | Model type: `"sd15"`, `"sdxl"`, `"flux"`, or `"auto"` |
| `r` | int | 16 | LoRA rank |
| `alpha` | float | None | LoRA alpha (defaults to r) |
| `lora_dropout` | float | 0.0 | Dropout probability |
| `target_modules` | list | None | Target modules (auto-detected if None) |

### Training Targets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_transformer` | bool | True | Train UNet (SD) or Transformer (Flux) |
| `train_text_encoder` | bool | False | Train CLIP text encoder |
| `train_t5` | bool | False | Flux only: Train T5 encoder (memory intensive) |

### Image Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | int | Auto | Training resolution (512 for SD1.5, 1024 for SDXL/Flux) |
| `center_crop` | bool | True | Use center crop |
| `random_flip` | bool | True | Random horizontal flip |
| `torch_dtype` | str | `"bfloat16"` | Data type (`"float16"`, `"bfloat16"`, `"float32"`) |

### Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `merge_lora` | bool | False | Merge LoRA into base model and save full model |
| `output_components` | list | None | Components to save when merge_lora=True: `"unet"`, `"text_encoder"`, `"text_encoder_2"`, `"vae"`, `"all"` |

When `merge_lora=False` (default), only LoRA adapter weights are saved. When `merge_lora=True`, the LoRA weights are merged into the base model and full model components are saved, enabling subsequent quantization or ONNX export.

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
| `mixed_precision` | `"bf16"` | Mixed precision mode |
| `lr_scheduler` | `"constant"` | LR scheduler type |
| `max_grad_norm` | 1.0 | Max gradient norm |
| `snr_gamma` | None | SNR gamma for Min-SNR weighting |

## Data Preprocessing

Use `SDLoRADataContainer` with `sd_lora_preprocess` for automatic data preprocessing.

### Preprocessing Chain

The preprocessing chain supports multiple steps:

| Step | Default | Description |
|------|---------|-------------|
| `image_filtering` | Disabled | Filter low quality images |
| `auto_caption` | Disabled | Generate captions with VLM |
| `auto_tagging` | Disabled | Generate tags with WD14 |
| `caption_tag_merge` | Disabled | Merge captions and tags |
| `image_resizing` | Disabled | Resize images |
| `aspect_ratio_bucketing` | Enabled | Group by aspect ratio |

### Auto Captioning

Automatically generate captions using vision-language models:

```json
{
    "data_configs": [
        {
            "name": "train_data",
            "type": "SDLoRADataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            },
            "pre_process_data_config": {
                "type": "sd_lora_preprocess",
                "params": {
                    "base_resolution": 1024,
                    "enable_steps": ["auto_caption", "aspect_ratio_bucketing"],
                    "step_params": {
                        "auto_caption": {
                            "model_type": "florence2",
                            "prefix": "a photo of sks"
                        }
                    }
                }
            }
        }
    ]
}
```

#### Supported Captioning Models

| Model | Default Model Name | Features |
|-------|-------------------|----------|
| `blip2` | `Salesforce/blip2-opt-2.7b` | General captions |
| `florence2` | `microsoft/Florence-2-large` | Detailed descriptions |

#### Florence-2 Tasks

| Task | Description |
|------|-------------|
| `<CAPTION>` | Brief caption |
| `<DETAILED_CAPTION>` | Detailed caption (default) |
| `<MORE_DETAILED_CAPTION>` | Very detailed caption |

#### Auto Caption Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `"blip2"` | Captioning model |
| `model_name` | None | Custom model path |
| `prefix` | `""` | Prefix for captions (e.g., trigger word) |
| `suffix` | `""` | Suffix for captions |
| `overwrite` | False | Overwrite existing .txt files |
| `device` | `"cuda"` | Device for inference |

### Aspect Ratio Bucketing

Groups images by aspect ratio to minimize padding and improve training quality:

```json
{
    "pre_process_data_config": {
        "type": "sd_lora_preprocess",
        "params": {
            "base_resolution": 1024,
            "bucket_mode": "sdxl",
            "enable_steps": ["aspect_ratio_bucketing"]
        }
    }
}
```

| `bucket_mode` | Resolution | Use Case |
|---------------|------------|----------|
| `"sd15"` | 512 | Stable Diffusion 1.5 |
| `"sdxl"` | 1024 | SDXL and Flux |
| `"auto"` | Auto | Auto-detect from base_resolution |

## Model-Specific Examples

### Stable Diffusion 1.5

```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "runwayml/stable-diffusion-v1-5"
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "SDLoRADataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            },
            "pre_process_data_config": {
                "type": "sd_lora_preprocess",
                "params": {
                    "base_resolution": 512,
                    "bucket_mode": "sd15"
                }
            }
        }
    ],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "r": 4,
            "resolution": 512
        }
    }
}
```

### SDXL

```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "stabilityai/stable-diffusion-xl-base-1.0"
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "SDLoRADataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            },
            "pre_process_data_config": {
                "type": "sd_lora_preprocess",
                "params": {
                    "base_resolution": 1024,
                    "bucket_mode": "sdxl"
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
        "type": "HfModel",
        "model_path": "black-forest-labs/FLUX.1-dev"
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "SDLoRADataContainer",
            "load_dataset_config": {
                "type": "image_folder_dataset",
                "params": {"data_dir": "train_images"}
            },
            "pre_process_data_config": {
                "type": "sd_lora_preprocess",
                "params": {
                    "base_resolution": 1024,
                    "bucket_mode": "sdxl"
                }
            }
        }
    ],
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "r": 32,
            "torch_dtype": "bfloat16",
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
pipe.load_lora_weights("output/unet_lora")
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

## Post-Training: Quantization and ONNX Export

After LoRA training, you can quantize or export the model to ONNX. Use `merge_lora=True` to output merged model components as a `CompositeModel`, which allows subsequent passes to automatically process each component.

### Single Config Pipeline

With `merge_lora=True`, SDLoRA returns a `CompositeModelHandler` containing each component as a separate `HfModelHandler`. Subsequent passes (like `OnnxConversion`, `OnnxQuantization`) will automatically iterate over each component.

```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "stabilityai/stable-diffusion-xl-base-1.0"
    },
    "data_configs": [
        {
            "name": "train_data",
            "type": "SDLoRADataContainer",
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
            "merge_lora": true,
            "output_components": ["unet", "vae", "text_encoder"]
        },
        "convert": {
            "type": "OnnxConversion"
        },
        "quantize": {
            "type": "OnnxQuantization",
            "quant_mode": "static",
            "quant_format": "QDQ"
        }
    },
    "output_dir": "output"
}
```

**Flow:**
```
SDLoRA (merge_lora=true)
    ↓
CompositeModel:
  - unet: HfModelHandler
  - vae: HfModelHandler
  - text_encoder: HfModelHandler
    ↓
OnnxConversion (auto applies to each component)
    ↓
CompositeModel:
  - unet: ONNXModelHandler
  - vae: ONNXModelHandler
  - text_encoder: ONNXModelHandler
    ↓
OnnxQuantization (auto applies to each component)
    ↓
CompositeModel:
  - unet: ONNXModelHandler (quantized)
  - vae: ONNXModelHandler (quantized)
  - text_encoder: ONNXModelHandler (quantized)
```

### Output Structure

```
output/
├── unet/           # Merged and optimized UNet
├── text_encoder/   # Text encoder
├── text_encoder_2/ # SDXL only
└── vae/            # VAE
```

### Flux Components

For Flux models, use `transformer` instead of `unet`:

```json
{
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "merge_lora": true,
            "output_components": ["transformer", "vae"]
        },
        "convert": {
            "type": "OnnxConversion"
        }
    }
}
```

### Export All Components

Use `"all"` to export all available components:

```json
{
    "passes": {
        "sd_lora": {
            "type": "SDLoRA",
            "train_data_config": "train_data",
            "merge_lora": true,
            "output_components": ["all"]
        }
    }
}
```

For SDXL, this exports: `unet`, `text_encoder`, `text_encoder_2`, `vae`
For Flux, this exports: `transformer`, `text_encoder`, `text_encoder_2`, `vae`

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
