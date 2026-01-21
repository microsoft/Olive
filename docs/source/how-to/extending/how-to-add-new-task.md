# How to Add a New Task or Diffusers Component for ONNX Export

This guide explains how to add IO configurations for a new HuggingFace task or diffusers component to enable ONNX model export.

Olive uses YAML-based IO configurations to define input/output specifications for ONNX export. These configurations specify tensor shapes, data types, and dynamic axes for each model input and output.

There are two types of configurations:
- **Task configs** (`tasks.yaml`): For HuggingFace transformers tasks like text-generation, text-classification, etc.
- **Diffusers component configs** (`diffusers.yaml`): For Stable Diffusion and similar diffusion model components like UNet, VAE, text encoders, etc.

## File Locations

IO config files are located in `olive/assets/io_configs/`:

```
olive/assets/io_configs/
├── tasks.yaml      # Task-based configurations
├── diffusers.yaml  # Diffusers component configurations
└── defaults.yaml   # Default dimension values and aliases
```

## Task-based IO Configs (`tasks.yaml`)

### Format

Each task defines its input/output specifications:

```yaml
task-name:
  inputs:
    input_name:
      shape: [dim1, dim2, ...]      # Shape template for dummy input generation
      axes: {0: axis_name, 1: ...}  # Dynamic axes for ONNX export
      dtype: int64 | float          # Data type (default: int64)
      max_value: vocab_size         # Optional: max value for random input
      optional: true                # Optional: skip if not in model.forward()
  outputs:
    output_name:
      axes: {0: axis_name, ...}     # Dynamic axes for ONNX export
  with_past:                        # Optional: overrides for KV cache scenarios
    input_name:
      shape: [...]
      axes: {...}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `shape` | List of dimension names or integers. Used to generate dummy inputs for ONNX export. Dimension names are resolved from model config or defaults. |
| `axes` | Dict mapping axis index to axis name. Defines which dimensions are dynamic in the exported ONNX model. |
| `dtype` | Data type: `int64`, `int32`, or `float`. Defaults to `int64` for inputs. |
| `optional` | If `true`, the input is only included if it exists in `model.forward()` signature. |
| `max_value` | Maximum value for random input generation (e.g., `vocab_size` for input_ids). |
| `with_past` | Alternative shapes/axes when using KV cache (`use_past_in_inputs=True`). |

### Example: Adding a New Task

To add support for a new task, add an entry to `tasks.yaml`:

```yaml
# Custom task for a new model type
my-custom-task:
  inputs:
    input_ids:
      shape: [batch_size, sequence_length]
      axes: {0: batch_size, 1: sequence_length}
      dtype: int64
      max_value: vocab_size
    attention_mask:
      shape: [batch_size, sequence_length]
      axes: {0: batch_size, 1: sequence_length}
      dtype: int64
    custom_input:
      shape: [batch_size, custom_dim]
      axes: {0: batch_size, 1: custom_dim}
      dtype: float
      optional: true
  outputs:
    logits:
      axes: {0: batch_size, 1: sequence_length, 2: vocab_size}
    custom_output:
      axes: {0: batch_size, 1: hidden_size}
```

### Supported Tasks

Currently supported tasks include:
- `text-generation`
- `text-classification`
- `feature-extraction`
- `fill-mask`
- `token-classification`
- `question-answering`
- `multiple-choice`
- `text2text-generation`
- `image-classification`
- `object-detection`
- `semantic-segmentation`
- `audio-classification`
- `automatic-speech-recognition`
- `zero-shot-image-classification`

## Diffusers Component Configs (`diffusers.yaml`)

### Format

Diffusers configurations define components and pipelines:

```yaml
components:
  component_name:
    inputs:
      input_name:
        shape: [dim1, dim2, ...]
        axes: {0: axis_name, ...}
        dtype: int64 | float
    outputs:
      output_name:
        axes: {0: axis_name, ...}
    sdxl_inputs:           # Optional: additional inputs for SDXL
      extra_input:
        shape: [...]
        axes: {...}
    optional_inputs:       # Optional: conditional inputs
      optional_input:
        shape: [...]
        axes: {...}
        condition: config_attr  # Only include if config.config_attr is True

pipelines:
  pipeline_name:
    - component_name
    - component_config:alias_name  # Use component_config with alias
```

### Example: Adding a New Diffusers Component

```yaml
components:
  my_custom_transformer:
    inputs:
      hidden_states:
        shape: [batch_size, in_channels, height, width]
        axes: {0: batch_size, 1: in_channels, 2: height, 3: width}
        dtype: float
      encoder_hidden_states:
        shape: [batch_size, sequence_length, hidden_size]
        axes: {0: batch_size, 1: sequence_length, 2: hidden_size}
        dtype: float
      timestep:
        shape: [batch_size]
        axes: {0: batch_size}
        dtype: float
    outputs:
      out_sample:
        axes: {0: batch_size, 1: in_channels, 2: height, 3: width}
    optional_inputs:
      guidance:
        shape: [batch_size]
        axes: {0: batch_size}
        dtype: float
        condition: guidance_embeds  # Only if config.guidance_embeds is True

pipelines:
  my_custom_pipeline:
    - text_encoder
    - my_custom_transformer:transformer
    - vae_encoder
    - vae_decoder
```

### Supported Diffusers Components

Currently supported components include:
- `text_encoder`, `text_encoder_with_projection`, `t5_encoder`, `gemma2_text_encoder`
- `unet`, `sd3_transformer`, `flux_transformer`, `sana_transformer`
- `vae_encoder`, `vae_decoder`, `dcae_encoder`, `dcae_decoder`

Supported pipelines: `sd`, `sdxl`, `sd3`, `flux`, `sana`

## Default Values (`defaults.yaml`)

The `defaults.yaml` file defines:
1. **Aliases**: Alternative attribute names for the same concept across different models
2. **Default dimensions**: Fallback values when dimensions can't be resolved from model config

### Aliases

Aliases help resolve config attributes that have different names across models:

```yaml
aliases:
  num_layers: [num_hidden_layers, n_layer, n_layers]
  hidden_size: [dim, d_model, n_embd]
  num_attention_heads: [num_heads, n_head, n_heads, encoder_attention_heads]
  num_kv_heads: [num_key_value_heads]
  height: [sample_size, image_size, vision_config.image_size]
  width: [sample_size, image_size, vision_config.image_size]
  num_channels: [in_channels, vision_config.num_channels]
```

### Default Dimensions

Default values used when dimensions can't be resolved from model config:

```yaml
batch_size: 2
sequence_length: 16
past_sequence_length: 16
vocab_size: 32000
height: 64
width: 64
num_channels: 3
```

### Adding New Defaults

If your model uses a dimension not already defined, add it to `defaults.yaml`:

```yaml
# Add new dimension for your model
my_custom_dim: 128

# Add aliases if the same concept has different names
aliases:
  my_custom_dim: [custom_dim, my_dim]
```

## Dimension Resolution

When generating dummy inputs, dimensions in `shape` are resolved in this order:

1. **Model config with aliases**: Check `config.attr_name` for each alias
2. **Computed dimensions**: Special dimensions like `height_latent = height // 8`
3. **Default values**: Fall back to values in `defaults.yaml`

## Usage in Olive Workflows

Once you've added your IO config, Olive will automatically use it during ONNX conversion.

### Task-based Models

For HuggingFace transformers models, specify the task in `HfModel`:

```yaml
# olive_config.yaml
input_model:
  type: HfModel
  model_path: my-model
  task: my-custom-task  # Uses the task config you defined

passes:
  conversion:
    type: OnnxConversion
```

### Diffusers Models

For diffusion models, use `DiffusersModel`. Olive automatically detects the pipeline type and exports all components using the IO configs defined in `diffusers.yaml`:

```yaml
# olive_config.yaml
input_model:
  type: DiffusersModel
  model_path: stabilityai/stable-diffusion-xl-base-1.0

passes:
  conversion:
    type: OnnxConversion
```

Olive will automatically:
1. Detect the pipeline type (e.g., `sdxl`)
2. Identify exportable components (text_encoder, text_encoder_2, unet, vae_encoder, vae_decoder)
3. Use the corresponding IO configs from `diffusers.yaml` for each component

## Testing Your Config

After adding a new IO config, verify it works:

```python
from olive.common.hf.io_config import get_io_config, generate_dummy_inputs

# Test task config
io_config = get_io_config("my-model-path", task="my-custom-task")
print(io_config["input_names"])
print(io_config["output_names"])
print(io_config["dynamic_axes"])

# Generate dummy inputs
dummy_inputs = generate_dummy_inputs("my-model-path", task="my-custom-task")
for name, tensor in dummy_inputs.items():
    print(f"{name}: {tensor.shape}")
```

For diffusers:

```python
from olive.common.hf.io_config import get_diffusers_io_config, generate_diffusers_dummy_inputs

# Test diffusers config
io_config = get_diffusers_io_config("my_custom_transformer", config)
print(io_config["input_names"])

# Generate dummy inputs
dummy_inputs = generate_diffusers_dummy_inputs("my_custom_transformer", config)
```
