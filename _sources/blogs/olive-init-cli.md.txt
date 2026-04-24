# Introducing `olive init`: An Interactive Wizard for Model Optimization

*Author: Xiaoyu Zhang*
*Created: 2026-04-06*

Getting started with AI model optimization can be overwhelming — choosing the right exporter, quantization algorithm, precision, and target hardware involves navigating a complex decision space. The new **`olive init`** command solves this with an interactive, step-by-step wizard that guides you from model selection to a ready-to-run optimization command.

---

## Why `olive init`?

Olive offers a powerful set of CLI commands — `optimize`, `quantize`, `capture-onnx-graph`, `finetune`, `diffusion-lora`, and more — each with many options. While this flexibility is great for experts, it can be daunting for newcomers. Common questions include:

- *Which command should I use?*
- *What exporter is best for my LLM?*
- *Which quantization algorithm should I pick?*
- *Do I need calibration data?*

**`olive init`** answers all of these by walking you through a guided wizard. It asks the right questions, provides sensible defaults, and generates the exact CLI command or JSON config you need.

---

## Quick Start

```bash
pip install olive-ai
olive init
```

That's it! The wizard launches in your terminal and walks you through every step.

---

## How It Works

The wizard follows a simple 4-step flow:

### Step 1: Choose Your Model Type

```
? What type of model do you want to optimize?
❯ PyTorch (HuggingFace or local)
  ONNX
  Diffusers (Stable Diffusion, SDXL, Flux, etc.)
```

### Step 2: Specify Your Model

Depending on the model type, you can provide:

- **HuggingFace model name** (e.g., `meta-llama/Llama-3.1-8B`)
- **Local directory path**
- **AzureML registry path**
- **PyTorch model with custom script**

### Step 3: Configure Your Workflow

This is where the wizard really shines. Based on your model type, it presents relevant operations and guides you through the configuration:

::::{tab-set}

:::{tab-item} PyTorch Models

**Available operations:**

| Operation | Description |
|-----------|-------------|
| Optimize | Export to ONNX + quantize + graph optimize (all-in-one) |
| Export to ONNX | Convert to ONNX format using Model Builder, Dynamo, or TorchScript |
| Quantize | Apply PyTorch quantization (RTN, GPTQ, AWQ, QuaRot, SpinQuant) |
| Fine-tune | LoRA or QLoRA fine-tuning on custom datasets |

For the **Optimize** operation, you can choose between:

- **Auto Mode** — Olive automatically selects the best passes for your target hardware and precision
- **Custom Mode** — Manually pick which operations to include (export, quantize, graph optimization) and configure each one

:::

:::{tab-item} ONNX Models

**Available operations:**

| Operation | Description |
|-----------|-------------|
| Optimize | Auto-select best passes for target hardware |
| Quantize | Static, dynamic, block-wise RTN, HQQ, or BnB quantization |
| Graph optimization | Apply ONNX graph-level optimizations |
| Convert precision | FP32 → FP16 conversion |
| Tune session params | Auto-tune ONNX Runtime inference parameters |

:::

:::{tab-item} Diffusers Models

**Available operations:**

| Operation | Description |
|-----------|-------------|
| Export to ONNX | Export diffusion pipeline for ONNX Runtime deployment |
| LoRA Training | Fine-tune with LoRA on custom images (DreamBooth supported) |

**Supported architectures:** SD 1.x/2.x, SDXL, SD3, Flux, Sana

:::

::::

### Step 4: Choose Your Output

```
? What would you like to do?
❯ Generate CLI command (copy and run later)
  Generate configuration file (JSON, for olive run)
  Run optimization now
```

You can generate the command to review first, save a reusable JSON config, or execute immediately.

---

## Examples

### Example 1: Optimize a HuggingFace LLM for CPU with INT4

```
$ olive init

Welcome to Olive Init! This wizard will help you optimize your model.

? What type of model do you want to optimize? PyTorch (HuggingFace or local)
? How would you like to specify your model? HuggingFace model name
? Model name or path: Qwen/Qwen2.5-0.5B-Instruct
? What do you want to do? Optimize model (export to ONNX + quantize + graph optimize)
? How would you like to configure optimization? Auto Mode (recommended)
? Select target device: CPU
? Select target precision: INT4 (smallest size, best for LLMs)
? Output directory: ./olive-output
? What would you like to do? Generate CLI command (copy and run later)

Generated command:

  olive optimize -m Qwen/Qwen2.5-0.5B-Instruct --provider CPUExecutionProvider --precision int4 -o ./olive-output
```

### Example 2: Quantize a PyTorch Model with GPTQ

```
$ olive init

? What type of model do you want to optimize? PyTorch (HuggingFace or local)
? How would you like to specify your model? HuggingFace model name
? Model name or path: meta-llama/Llama-3.1-8B
? What do you want to do? Quantize only (PyTorch quantization)
? Select quantization algorithm: GPTQ - High quality, requires calibration
? Precision: int4
? Calibration data source: Use default (wikitext-2)
? Output directory: ./olive-output
? What would you like to do? Run optimization now
```

### Example 3: Fine-tune with LoRA

```
$ olive init

? What type of model do you want to optimize? PyTorch (HuggingFace or local)
? How would you like to specify your model? HuggingFace model name
? Model name or path: microsoft/Phi-4-mini-instruct
? What do you want to do? Fine-tune model (LoRA, QLoRA)
? Select fine-tuning method: LoRA (recommended)
? LoRA rank (r): 64 (default)
? LoRA alpha: 16
? Training dataset: HuggingFace dataset
? Dataset name: tatsu-lab/alpaca
? Train split: train
? How to construct training text? Use chat template
? Max sequence length: 1024
? Max training samples: 256
? Torch dtype for training: bfloat16 (recommended)
? Output directory: ./olive-output
? What would you like to do? Generate CLI command (copy and run later)

Generated command:

  olive finetune -m microsoft/Phi-4-mini-instruct --method lora --lora_r 64 --lora_alpha 16 -d tatsu-lab/alpaca --train_split train --use_chat_template --max_seq_len 1024 --max_samples 256 --torch_dtype bfloat16 -o ./olive-output
```

### Example 4: Train a Diffusion LoRA with DreamBooth

```
$ olive init

? What type of model do you want to optimize? Diffusers (Stable Diffusion, SDXL, Flux, etc.)
? Select diffuser model variant: Stable Diffusion XL (SDXL)
? Enter model name or path: stabilityai/stable-diffusion-xl-base-1.0
? What do you want to do? LoRA Training (fine-tune on custom images)
? LoRA rank (r): 16 (recommended)
? LoRA alpha: 16
? Training data source: Local image folder
? Path to image folder: ./my-dog-photos
? Enable DreamBooth training? Yes
? Instance prompt: a photo of sks dog
? Enable prior preservation? Yes
? Class prompt: a photo of a dog
? Max training steps: 1000 (recommended)
? Output directory: ./olive-output
? What would you like to do? Generate CLI command (copy and run later)

Generated command:

  olive diffusion-lora -m stabilityai/stable-diffusion-xl-base-1.0 --model_variant sdxl -r 16 --alpha 16 --lora_dropout 0.0 -d ./my-dog-photos --dreambooth --instance_prompt "a photo of sks dog" --with_prior_preservation --class_prompt "a photo of a dog" --num_class_images 200 --max_train_steps 1000 --learning_rate 1e-4 --train_batch_size 1 --gradient_accumulation_steps 4 --mixed_precision bf16 --lr_scheduler constant --lr_warmup_steps 0 -o ./olive-output
```

---

## Related Resources

- [Olive CLI Reference](https://microsoft.github.io/Olive/reference/cli.html)
- [Olive Getting Started Guide](https://microsoft.github.io/Olive/getting-started/getting-started.html)
- [Olive GitHub Repository](https://github.com/microsoft/Olive)
