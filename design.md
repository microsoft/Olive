# Olive Init CLI Wizard Design Document

## Table of Contents
1. [Background](#background)
2. [Goals and Non-Goals](#goals-and-non-goals)
3. [Functional Requirements](#functional-requirements)
4. [`olive init` Wizard Flow — Detailed Step-by-Step Design](#olive-init-wizard-flow--detailed-step-by-step-design)
5. [Architecture](#architecture)
6. [UI/UX Design](#uiux-design)
7. [Examples](#examples)
8. [Testing Strategy](#testing-strategy)
9. [Appendix](#appendix)

---

## Background

### Current State

Olive currently supports two usage modes:
- **Configuration file-based workflow**: Users write JSON configuration files specifying passes and parameters
- **Command-line interface (CLI)**: Direct commands for specific operations

**CLI Complexity:**
- 16 CLI commands (`run`, `optimize`, `quantize`, `finetune`, `capture-onnx`, `diffusion-lora`, etc.)
- Average 15-20 subcommands/arguments per command

**Configuration File Complexity:**
- Required sections: `input_model`, `passes`
- Optional sections: `data_configs`, `evaluators`, `search_strategy`, `auto_optimizer_config`, `systems`, `output_dir`, `host`, `target`
- 68 registered passes with individual config schemas
- 12+ execution providers (CPU, CUDA, QNN, VitisAI, OpenVINO, DML, WebGPU, etc.)
- 9+ quantization algorithms (RTN, GPTQ, AWQ, HQQ, QuaRot, SpinQuant, etc.)

**Supported Model Types (10 types):**
- `HFModel` - HuggingFace Transformers models
- `PyTorchModel` - Generic PyTorch models with custom loader
- `ONNXModel` - ONNX format models
- `DiffusersModel` - Stable Diffusion, SDXL, SD3, Flux, Sana models
- `OpenVINOModel` - Intel OpenVINO IR format models
- `QNNModel` - Qualcomm QNN compiled models
- `TensorFlowModel` - TensorFlow SavedModel format
- `CompositeModel` - Multi-component models (e.g., encoder-decoder)
- `DistributedHfModel` - Distributed HuggingFace models
- `DistributedOnnxModel` - Distributed ONNX models

### Problem Statement

This complexity creates significant barriers for new users:

1. **Discovery Challenge**: Users must browse documentation or use `--help` extensively to find relevant commands
2. **Decision Paralysis**: Many options make it difficult to know which passes and parameters to use
3. **Learning Curve**: Understanding the relationship between model types, execution providers, and optimization passes requires deep knowledge
4. **Error-Prone Configuration**: Manual configuration file creation is tedious and error-prone

### Proposed Solution

Introduce `olive init`, an interactive CLI wizard inspired by tools like `npm init`, `poetry init`, and `cargo init`. This wizard will guide users step-by-step through:
- Selecting their model type and source
- Choosing optimization operations
- Configuring target hardware and execution providers
- Generating and optionally executing the optimized workflow

---

## Goals and Non-Goals

### Goals

1. **Lower Onboarding Barrier**: Enable new users to start using Olive within minutes
2. **Guided Discovery**: Help users discover relevant commands, passes, and parameters
3. **Intelligent Defaults**: Provide smart defaults based on user selections
4. **Flexibility**: Support both "smart mode" (automated) and "detail mode" (manual selection)
5. **Generate Artifacts**: Produce runnable commands or configuration files
6. **Educational**: Teach users about Olive concepts as they make selections

### Non-Goals

1. **Replace Existing CLI**: `olive init` complements, not replaces, existing commands
2. **Cover All Passes**: Focus on common workflows, not every possible pass combination
3. **GUI Interface**: This is a terminal-based interactive wizard, not a graphical tool
4. **Runtime Optimization**: The wizard generates configurations; actual optimization runs separately (unless user opts to run immediately)

---

## Functional Requirements

### FR1: Model Type Selection

**What**: The wizard must identify the user's input model type and present relevant optimization operations based on that type.

**Why**: Different model types (PyTorch, ONNX, Diffusers) support different operations and passes. The wizard needs to filter and show only applicable options to avoid confusing users.

**Supported model types and their operations**:

#### PyTorch Models (HuggingFace/Local)
- **Export to ONNX**: Convert PyTorch model to ONNX format for deployment
- **Quantize**: Reduce model size and improve inference speed using algorithms like GPTQ, RTN, AWQ
- **Export + Quantize**: Combined workflow that exports and quantizes in one pipeline
- **Fine-tune**: Adapt pre-trained models using LoRA, QLoRA, or DoRA techniques

#### ONNX Models
- **Quantize**: Apply INT4, INT8, INT16, or FP16 quantization to reduce model size
- **Optimize**: Apply graph-level optimizations like operator fusion and constant folding

#### Diffuser Models
- **Export to ONNX**: Convert Stable Diffusion/SDXL/Flux models to ONNX format
- **LoRA Training**: Train custom LoRA adapters for image generation

---

### FR2: Dual Mode Support

**What**: The wizard must support two configuration modes - Auto Mode for beginners and Custom Mode for advanced users.

**Why**: New users want quick results without understanding every parameter, while advanced users need fine-grained control over the optimization pipeline.

**Auto Mode**:
- User only specifies: target device (CPU/GPU/NPU), execution provider, and desired precision
- Olive automatically selects the optimal combination of passes
- Best for: Users who want quick optimization without deep Olive knowledge

**Custom Mode**:
- User manually selects specific operations (export, quantize, fine-tune, etc.)
- User configures individual pass parameters (algorithm, precision, calibration data, etc.)
- Best for: Users who need precise control over the optimization pipeline

---

### FR3: Hardware Targeting

**What**: The wizard must automatically recommend appropriate passes based on the user's target hardware and execution provider.

**Why**: Different hardware platforms require different optimization strategies. For example, Qualcomm NPU needs QNN-specific passes, while Intel NPU needs OpenVINO passes. Users shouldn't need to know these details.

**Hardware-to-pass mapping examples**:
- **CPU** → Static quantization, transformer optimizations
- **NVIDIA GPU** → FP16 conversion, TensorRT integration
- **Qualcomm NPU** → QNN conversion, context binary generation
- **Intel NPU** → OpenVINO conversion and quantization
- **AMD NPU** → Vitis AI metadata and optimization

---

### FR4: Output Options

**What**: The wizard must provide flexible output options for how the user wants to use the generated configuration.

**Why**: Different users have different workflows - some want to review before running, some want immediate execution, and some need config files for CI/CD integration.

**Output options**:
- **Generate Command**: Display the equivalent `olive` CLI command that users can copy and run later
- **Generate Config File**: Save a JSON configuration file for use with `olive run`
- **Run Directly**: Execute the optimization immediately after wizard completion
- **Dry Run**: Preview what would be executed without actually running (for verification)

---

## `olive init` Wizard Flow — Detailed Step-by-Step Design

本节按照 wizard 的实际交互步骤，逐步描述用户看到什么、选了之后下一步是什么、最终生成什么。

---

### Step 1: Model Type Selection

```
$ olive init

Welcome to Olive Init! This wizard will help you optimize your model.

? What type of model do you want to optimize?
  ❯ PyTorch (HuggingFace or local)
    ONNX
    Diffuser (Stable Diffusion, SDXL, Flux, etc.)
```

用户选择后，后续所有步骤的可选项会根据 model type 不同而不同。

---

### Step 2: Model Source

根据 Step 1 选择的 model type，提示不同的来源方式：

**如果选了 PyTorch:**
```
? How would you like to specify your model?
  ❯ HuggingFace model name (e.g., meta-llama/Llama-3.1-8B)
    Local directory path
    AzureML registry path
    PyTorch model with custom script

? Model name or path: meta-llama/Llama-3.1-8B
```

如果选了 "PyTorch model with custom script"，则额外问：
```
? Path to model script (.py): ./my_model.py
? Script directory (optional): ./scripts/
```

> 脚本中需定义 `_model_loader()`, `_io_config()` 或 `_dummy_inputs()` 等函数。

**如果选了 ONNX:**
```
? Enter ONNX model path (file or directory):
  [./model.onnx]
```

**如果选了 Diffuser:**
```
? Select diffuser model variant:
  ❯ Auto-detect
    Stable Diffusion (SD 1.x/2.x)
    Stable Diffusion XL (SDXL)
    Stable Diffusion 3 (SD3)
    Flux
    Sana

? Enter model name or path: stabilityai/stable-diffusion-xl-base-1.0
```

---

### Step 3: Operation Selection

这是最核心的步骤——不同 model type 展示不同的可用操作。

---

#### 3A. PyTorch Model → Operation Selection

```
? What do you want to do?
  ❯ Optimize model (export to ONNX + quantize + graph optimize)
    Export to ONNX only
    Quantize only (PyTorch quantization)
    Fine-tune model (LoRA, QLoRA)
```

##### 选了 "Optimize model" → 进入 Auto/Custom Mode

```
? How would you like to configure optimization?
  ❯ Auto Mode (recommended) - Automatically select best passes for your target
    Custom Mode - Manually pick operations and parameters
```

**Auto Mode 流程:**

```
? Select target device:
  ❯ CPU
    GPU (NVIDIA CUDA)
    GPU (NvTensorRTRTX)
    NPU (Qualcomm QNN)
    NPU (Intel OpenVINO)
    NPU (AMD Vitis AI)
    WebGPU

? Select target precision:
  ❯ INT4 (smallest size, best for LLMs)
    INT8 (balanced)
    FP16 (half precision)
    FP32 (full precision)
```

Wizard 根据 (device, precision) 自动决定 pass 流水线，用户无需了解细节：

| Device | Precision | 自动生成的 Pass 流水线 | 生成命令 |
|---|---|---|---|
| CPU | int4 | `Gptq` → `ModelBuilder(int4)` | `olive optimize --provider CPU --precision int4` |
| CPU | int8 | `ModelBuilder` → `OnnxStaticQuantization` | `olive optimize --provider CPU --precision int8` |
| CPU | fp16 | `ModelBuilder` → `OnnxFloatToFloat16` | `olive optimize --provider CPU --precision fp16` |
| CPU | fp32 | `ModelBuilder` | `olive optimize --provider CPU --precision fp32` |
| GPU (CUDA) | int4 | `Gptq` → `ModelBuilder(int4)` | `olive optimize --provider CUDA --precision int4` |
| GPU (CUDA) | fp16 | `ModelBuilder` → `OnnxFloatToFloat16` | `olive optimize --provider CUDA --precision fp16` |
| NPU (QNN) | int4 | `QuaRot` → `Gptq` → `ModelBuilder` → `DynamicToFixedShape` → `StaticLLM` | `olive optimize --provider QNN --precision int4` |
| NPU (OpenVINO) | 任意 | `OpenVINOOptimumConversion` → `OpenVINOIoUpdate` → `OpenVINOEncapsulation` | `olive optimize --provider OpenVINO --precision ...` |
| NPU (VitisAI) | int4 | `QuaRot` → `Gptq` → `ModelBuilder` → `StaticLLM` → `VitisAIAddMetaData` | `olive optimize --provider VitisAI --precision int4` |
| WebGPU | int4 | `Gptq` → `ModelBuilder(int4)` → `OnnxIODataTypeConverter` | `olive optimize --provider WebGpu --precision int4` |
| NvTensorRTRTX | int4 | `Gptq` → `ModelBuilder(int4)` | `olive optimize --provider NvTensorRTRTX --precision int4` |

→ 直接跳到 [Step 4: Output](#step-4-output)

**Custom Mode 流程:**

```
? Select operations to perform: (multi-select)
  ◉ Export to ONNX
  ◉ Quantize
  ◯ Graph Optimization
```

如果选了 **Export to ONNX**:
```
? Select exporter:
  ❯ Model Builder (recommended for LLMs)
    Dynamo Exporter (general purpose)
    TorchScript Exporter (legacy)

? Export precision:
  ❯ fp16
    fp32
    bf16
    int4
```
如果 precision=int4:
```
? INT4 block size:
  ❯ 32 (recommended)
    16 / 64 / 128 / 256
```

如果选了 **Quantize**:
```
? Select quantization algorithm:
  ❯ RTN - Round-to-Nearest, fast, no calibration needed
    GPTQ - High quality, requires calibration data
    AWQ - Activation-aware, good for LLMs
    QuaRot - Rotation-based, for QNN/VitisAI deployment
    SpinQuant - Spin quantization

? Quantization precision:
  ❯ int4
    uint4
    int8
```

如果算法需要 calibration data (GPTQ / AWQ / QuaRot / SpinQuant):
```
? Calibration data source:
  ❯ Use default (wikitext-2)
    HuggingFace dataset
    Local file

# 如果选 HuggingFace dataset:
? Dataset name: [wikitext]
? Subset (optional): [wikitext-2-raw-v1]
? Split: [train]
? Number of samples: [128]
```

如果选了 **Graph Optimization**:
```
? Select optimizations: (multi-select)
  ◉ Peephole optimization
  ◉ Transformer optimization
```

→ 选择 target device / provider → 进入 [Step 4: Output](#step-4-output)

---

##### 选了 "Export to ONNX only"

```
? Select exporter:
  ❯ Model Builder (recommended for LLMs)
    Dynamo Exporter
    TorchScript Exporter

? Export precision:
  ❯ fp16
    fp32 / bf16 / int4
```

如果选了 Model Builder + int4:
```
? INT4 block size: [32]
? INT4 accuracy level: (1=fp32 / 2=fp16 / 3=bf16 / 4=int8): [4]
```

如果选了 Dynamo Exporter:
```
? Target opset version: [17]
? Torch dtype (optional): fp32 / fp16
```

**生成命令:** `olive capture-onnx-graph -m ... --use_model_builder --precision fp16`

→ 进入 [Step 4: Output](#step-4-output)

---

##### 选了 "Quantize only"

```
? Select quantization algorithm:
  ❯ RTN - Fast, no calibration needed
    GPTQ - High quality, requires calibration
    AWQ - Activation-aware
    QuaRot - For QNN/VitisAI deployment
    SpinQuant - Spin quantization

? Precision:
  ❯ int4
    uint4
    int8
```

如果算法需要 calibration (GPTQ, AWQ, QuaRot, SpinQuant):
```
? Calibration data source:
  ❯ Skip (use default)
    HuggingFace dataset
    Local file
```

算法与 pass 的对应关系:

| 用户选择 | 实际 Pass |
|---|---|
| RTN | `Rtn` |
| GPTQ | `Gptq` (olive 实现) 或 `GptqQuantizer` (autogptq 实现) |
| AWQ | `AutoAWQQuantizer` |
| QuaRot | `QuaRot` |
| SpinQuant | `SpinQuant` |

**生成命令:** `olive quantize -m ... --algorithm rtn --precision int4`

→ 进入 [Step 4: Output](#step-4-output)

---

##### 选了 "Fine-tune model"

```
? Select fine-tuning method:
  ❯ LoRA (recommended)
    QLoRA (quantized, saves GPU memory)

? LoRA rank (r):
  ❯ 64 (default)
    4 / 8 / 16 / 32

? LoRA alpha: [16]
```

```
? Training dataset:
  ❯ HuggingFace dataset
    Local file

? Dataset name: tatsu-lab/alpaca
? Train split: [train]
? Eval split (optional, press enter to skip): []
```

```
? How to construct training text?
  ❯ Single text field (specify column name)
    Text template (e.g., '### Question: {prompt} \n### Answer: {response}')
    Use chat template

? Text field name: text
? Max sequence length: [1024]
? Max training samples: [256]
```

```
? Torch dtype for training:
  ❯ bfloat16 (recommended)
    float16
    float32
```

**生成命令:** `olive finetune -m ... --method lora -d tatsu-lab/alpaca --lora_r 64`
**使用的 Pass:** `LoRA` 或 `QLoRA`

→ 进入 [Step 4: Output](#step-4-output)

---

#### 3B. ONNX Model → Operation Selection

```
? What do you want to do?
  ❯ Optimize model (auto-select best passes for target hardware)
    Quantize
    Graph optimization
    Convert precision (FP32 → FP16)
    Tune session parameters
```

##### 选了 "Optimize model" → Auto Mode

```
? Select target device:
  ❯ CPU
    GPU (NVIDIA CUDA)
    NPU (Qualcomm QNN)
    NPU (Intel OpenVINO)
    NPU (AMD Vitis AI)
    WebGPU

? Select target precision:
  ❯ INT4
    INT8
    FP16
    FP32
```

自动选择的 pass 流水线 (ONNX 模型不需要 export，直接优化):

| Device | Precision | 自动选择的 Pass 流水线 |
|---|---|---|
| CPU | int4 | `OnnxPeepholeOptimizer` → `OnnxBlockWiseRtnQuantization` |
| CPU | int8 | `OnnxPeepholeOptimizer` → `OnnxStaticQuantization` |
| CPU | fp16 | `OnnxPeepholeOptimizer` → `OnnxFloatToFloat16` |
| CPU | fp32 | `OnnxPeepholeOptimizer` |
| QNN | int4 | `DynamicToFixedShape` → `OnnxBlockWiseRtnQuantization` → `StaticLLM` |
| VitisAI | int4 | `OnnxBlockWiseRtnQuantization` → `StaticLLM` → `VitisAIAddMetaData` |
| WebGPU | int4 | `OnnxPeepholeOptimizer` → `OnnxBlockWiseRtnQuantization` → `OnnxIODataTypeConverter` |

**生成命令:** `olive optimize -m ./model.onnx --provider CPU --precision int4`

→ 进入 [Step 4: Output](#step-4-output)

---

##### 选了 "Quantize"

```
? Select quantization type:
  ❯ Static Quantization (INT8) - requires calibration data
    Dynamic Quantization (INT8) - no calibration needed
    Block-wise RTN (INT4) - no calibration needed
    HQQ Quantization (INT4) - no calibration needed
    BnB Quantization (FP4/NF4) - no calibration needed
```

对应关系:

| 用户选择 | 实际 Pass | 需要 Calibration |
|---|---|---|
| Static Quantization (INT8) | `OnnxStaticQuantization` | 是 |
| Dynamic Quantization (INT8) | `OnnxDynamicQuantization` | 否 |
| Block-wise RTN (INT4) | `OnnxBlockWiseRtnQuantization` | 否 |
| HQQ Quantization (INT4) | `OnnxHqqQuantization` | 否 |
| BnB Quantization (FP4/NF4) | `OnnxBnB4Quantization` | 否 |

如果选了 Static Quantization:
```
? Calibration data source:
  ❯ HuggingFace dataset
    Local file

? Dataset name: Salesforce/wikitext
? Subset: wikitext-2-raw-v1
? Split: train
? Number of samples: [128]
```

**生成命令:** `olive quantize -m ./model.onnx --implementation ort --precision int8`

→ 进入 [Step 4: Output](#step-4-output)

---

##### 选了 "Graph optimization"

```
? Select optimizations: (multi-select)
  ◉ Peephole optimization (constant folding, dead code elimination)
  ◉ Transformer optimization (operator fusion for transformers)
```


**生成命令:** `olive optimize -m ./model.onnx --precision fp32` 或多个 `olive run-pass`

→ 进入 [Step 4: Output](#step-4-output)

---

##### 选了 "Convert precision (FP32 → FP16)"

无需额外参数。

**生成命令:** `olive run-pass --pass-name OnnxFloatToFloat16 -m ./model.onnx`

→ 进入 [Step 4: Output](#step-4-output)

---

##### 选了 "Tune session parameters"

```
? Select target device:
  ❯ CPU
    GPU

? Select execution providers: (multi-select)
  ◉ CPUExecutionProvider
  ◯ CUDAExecutionProvider
  ◯ TensorrtExecutionProvider

? CPU cores for thread tuning (optional, press enter to skip): []
? Enable IO binding? (y/N): N
? Enable CUDA graph? (y/N): N
```

**生成命令:** `olive tune-session-params -m ./model.onnx --device cpu`

→ 进入 [Step 4: Output](#step-4-output)

---

#### 3C. Diffuser Model → Operation Selection

```
? What do you want to do?
  ❯ Export to ONNX (for deployment with ONNX Runtime)
    LoRA Training (fine-tune on custom images)
```

##### 选了 "Export to ONNX"

```
? Conversion device:
  ❯ CPU
    GPU

? Torch dtype:
  ❯ float16
    float32

? Target ONNX opset: [17]
```

> 注意: Diffuser 模型只支持 PyTorch Exporter (OnnxConversion)，不支持 Model Builder。

**生成命令:** `olive capture-onnx-graph -m stabilityai/stable-diffusion-xl-base-1.0 --torch_dtype float16`

→ 进入 [Step 4: Output](#step-4-output)

---

##### 选了 "LoRA Training"

**LoRA 参数:**
```
? LoRA rank (r):
  ❯ 16 (recommended)
    4 / 8 / 32 / 64

? LoRA alpha (default = same as rank): [16]
? LoRA dropout: [0.0]
```

**数据来源:**
```
? Training data source:
  ❯ Local image folder
    HuggingFace dataset
```

如果选了 Local image folder:
```
? Path to image folder: ./training-images/
```

如果选了 HuggingFace dataset:
```
? Dataset name: linoyts/Tuxemon
? Split: [train]
? Image column name: [image]
? Caption column name (optional): [text]
```

**DreamBooth (可选):**
```
? Enable DreamBooth training? (y/N): y

? Instance prompt (e.g., 'a photo of sks dog'): a photo of sks dog
? Enable prior preservation? (y/N): y
? Class prompt (e.g., 'a photo of a dog'): a photo of a dog
? Class data directory (optional): []
? Number of class images: [200]
```

**训练参数:**
```
? Max training steps:
  ❯ 1000 (recommended)
    500 (quick) / 2000 (thorough) / Custom

? Learning rate: [1e-4]
? Train batch size: [1]
? Gradient accumulation steps: [4]

? Mixed precision:
  ❯ bf16 (recommended)
    fp16
    no

? Learning rate scheduler:
  ❯ constant
    linear / cosine / cosine_with_restarts / polynomial / constant_with_warmup

? Warmup steps: [0]
? Random seed (optional): []
```

如果 model variant 是 Flux:
```
? Guidance scale (Flux-specific): [3.5]
```

```
? Merge LoRA into base model? (y/N): N
```

**生成命令:**
```
olive diffusion-lora \
    -m stabilityai/stable-diffusion-xl-base-1.0 \
    --model_variant sdxl \
    -r 16 --alpha 16 \
    -d ./training-images/ \
    --max_train_steps 1000 \
    --learning_rate 1e-4 \
    --mixed_precision bf16
```
**使用的 Pass:** `SDLoRA`

→ 进入 [Step 4: Output](#step-4-output)

---

### Step 4: Output

所有路径最终汇聚到这里。

```
? Output directory: [./olive-output]

? What would you like to do?
  ❯ Generate CLI command (copy and run later)
    Generate configuration file (JSON, for olive run)
    Run optimization now
    Generate both and run
```

##### 如果选了 "Generate CLI command":

```
Generated command:

  olive optimize \
      -m meta-llama/Llama-3.1-8B \
      --provider CPUExecutionProvider \
      --precision int4 \
      -o ./olive-output

? Run this command now? (Y/n)
```

##### 如果选了 "Generate configuration file":

```
Configuration saved to: ./olive-output/olive-config.json

You can run it later with:
  olive run --config ./olive-output/olive-config.json
```

##### 如果选了 "Run optimization now":

```

Running optimization...
<actual olive CLI execution>

Optimization complete! Output saved to: ./olive-output
```

---

## Architecture

### File Structure

```
olive/cli/
├── __init__.py
├── init.py                    # Main init command
├── init/
│   ├── __init__.py
│   ├── wizard.py              # Interactive wizard logic
│   ├── prompts.py             # Question definitions and prompts
│   ├── validators.py          # Input validation
│   ├── templates.py           # Command/config templates
│   ├── pass_resolver.py       # Smart mode pass selection logic
│   └── models/
│       ├── __init__.py
│       ├── pytorch.py         # PyTorch-specific flows
│       ├── onnx.py            # ONNX-specific flows
│       └── diffuser.py        # Diffuser-specific flows
```

### Class Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         InitCommand                                  │
│  (extends BaseOliveCLICommand)                                      │
├─────────────────────────────────────────────────────────────────────┤
│ + register_subcommand(subparsers)                                   │
│ + run(args)                                                         │
│ - _start_wizard(args)                                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         InitWizard                                   │
├─────────────────────────────────────────────────────────────────────┤
│ - model_type: ModelType                                             │
│ - model_path: str                                                   │
│ - mode: WizardMode                                                  │
│ - target_device: Device                                             │
│ - target_provider: ExecutionProvider                                │
│ - target_precision: Precision                                       │
│ - operations: List[Operation]                                       │
│ - config: Dict                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ + start()                                                           │
│ + prompt_model_type()                                               │
│ + prompt_model_source()                                             │
│ + prompt_mode()                                                     │
│ + prompt_smart_mode_config()                                        │
│ + prompt_detail_mode_config()                                       │
│ + prompt_additional_options()                                       │
│ + prompt_output_action()                                            │
│ + generate_command() -> str                                         │
│ + generate_config() -> Dict                                         │
│ + execute()                                                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
          ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
          │ PyTorchFlow │  │  OnnxFlow   │  │DiffuserFlow │
          ├─────────────┤  ├─────────────┤  ├─────────────┤
          │get_operations│ │get_operations│ │get_operations│
          │get_passes()  │ │get_passes()  │ │get_passes()  │
          │get_params()  │ │get_params()  │ │get_params()  │
          └─────────────┘  └─────────────┘  └─────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PassResolver                                  │
├─────────────────────────────────────────────────────────────────────┤
│ + resolve_passes(model_type, provider, precision, operations)       │
│ + get_pass_config(pass_name, user_params)                          │
│ + validate_pass_compatibility(passes, provider)                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Integration with Existing CLI

```
olive/cli/launcher.py
    │
    ├── register_command(RunCommand)
    ├── register_command(OptimizeCommand)
    ├── register_command(QuantizeCommand)
    ├── register_command(FinetuneCommand)
    ├── ...
    └── register_command(InitCommand)  # New
```

---

## UI/UX Design

### Design Principles

1. **Progressive Disclosure**: Show only relevant options at each step
2. **Smart Defaults**: Pre-select common/recommended options
3. **Clear Language**: Avoid jargon; explain technical terms
4. **Recoverable Errors**: Allow users to go back and change selections
5. **Visual Hierarchy**: Use consistent formatting and indicators

### Visual Elements

```
# Selection indicator
  ❯ Selected option (highlighted)
    Other option

# Checkbox
  ◉ Selected
  ◯ Not selected

# Progress indicator
Step 2/6: Model Source

# Section dividers
════════════════════════════════════════════════════════════

# Code blocks
┌────────────────────────────────────────────────────────────┐
│ olive optimize -m meta-llama/Llama-3.1-8B \                │
│     --provider CPUExecutionProvider \                      │
│     --precision int4                                       │
└────────────────────────────────────────────────────────────┘

# Help text
? Select target precision:
  ❯ INT4 (smallest size, best for LLMs)
    INT8 (balanced size and accuracy)
    │
    └─ Tip: INT4 reduces model size by ~75% with minimal accuracy loss
```

### Keyboard Navigation

| Key | Action |
|-----|--------|
| ↑/↓ | Navigate options |
| Enter | Select/Confirm |
| Space | Toggle checkbox |
| Ctrl+C | Cancel wizard |
| ? | Show help for current step |

### Error Handling

```
? Enter model path: /invalid/path
  ✗ Path does not exist. Please enter a valid path.

? Enter model path: /path/to/model
  ✓ Model found: LlamaForCausalLM
```

---

## Examples

### Example 1: Quick Start - Auto Mode

```bash
$ olive init

Welcome to Olive Init! This wizard will help you optimize your model.

? What type of model do you want to optimize? PyTorch (HuggingFace or local)
? Enter model name or path: microsoft/phi-2
? How would you like to configure? Auto Mode (recommended)
? Select target device: CPU
? Select target precision: INT4
? Output directory: ./phi2-optimized
? What would you like to do? Generate CLI command

Generated command:
  olive optimize -m microsoft/phi-2 \
      --provider CPUExecutionProvider \
      --precision int4 \
      --output ./phi2-optimized

? Run this command now? Yes

Running optimization...
<run olive CLI here>

Optimization complete! Output saved to: ./phi2-optimized
```

### Example 2: Custom Mode - ONNX Quantization

```bash
$ olive init

? What type of model do you want to optimize? ONNX
? Enter model path: ./models/bert.onnx
? How would you like to configure? Custom Mode
? Select operations:
  ◉ Quantize model
  ◉ Graph optimization
? Select quantization algorithm: Static Quantization (INT8)
? Calibration data source: HuggingFace dataset
? HuggingFace dataset name: glue/mrpc
? Output directory: ./bert-quantized
? What would you like to do? Generate configuration file

Configuration saved to: ./bert-quantized/olive-config.json
```

### Example 3: Diffuser Model - LoRA Training

```bash
$ olive init

? What type of model do you want to optimize? Diffuser
? Select diffuser variant: Stable Diffusion XL
? Enter model name: stabilityai/stable-diffusion-xl-base-1.0
? How would you like to configure? Custom Mode
? Select operations:
  ◉ LoRA Training
? Training dataset: Local file
? Dataset path: ./training-images/
? Output directory: ./sdxl-lora
? What would you like to do? Run optimization now

Running LoRA training...
```

### Example 4: Non-Interactive Mode

```bash
# For CI/CD pipelines or scripting
$ olive init \
    --model-type pytorch \
    --model meta-llama/Llama-3.1-8B \
    --smart \
    --device gpu \
    --provider CUDAExecutionProvider \
    --precision int4 \
    --output-dir ./llama-optimized \
    --no-interactive
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit_tests/cli/test_init.py

import pytest
from olive.cli.init.wizard import InitWizard, ModelType, WizardMode
from olive.cli.init.pass_resolver import PassResolver

class TestPassResolver:
    def test_resolve_pytorch_int4_cpu(self):
        resolver = PassResolver()
        passes = resolver.resolve_passes(
            model_type="pytorch",
            provider="CPUExecutionProvider",
            precision="int4",
            operations=[]
        )

        assert "builder" in passes or "conversion" in passes
        assert "quantization" in passes

    def test_resolve_onnx_int8_cuda(self):
        resolver = PassResolver()
        passes = resolver.resolve_passes(
            model_type="onnx",
            provider="CUDAExecutionProvider",
            precision="int8",
            operations=[]
        )

        assert "quantization" in passes
        assert passes["quantization"]["type"] == "OnnxStaticQuantization"

    def test_resolve_openvino_provider(self):
        resolver = PassResolver()
        passes = resolver.resolve_passes(
            model_type="pytorch",
            provider="OpenVINOExecutionProvider",
            precision="int8",
            operations=[]
        )

        assert "conversion" in passes
        assert passes["conversion"]["type"] == "OpenVINOOptimumConversion"

class TestInitWizard:
    def test_generate_smart_command(self, mock_wizard):
        mock_wizard.mode = WizardMode.SMART
        mock_wizard.model_path = "microsoft/phi-2"
        mock_wizard.target_provider = "CPUExecutionProvider"
        mock_wizard.target_precision = "int4"
        mock_wizard.output_dir = "./output"

        command = mock_wizard.generate_command()

        assert "olive optimize" in command
        assert "-m microsoft/phi-2" in command
        assert "--provider CPUExecutionProvider" in command
        assert "--precision int4" in command

    def test_generate_config(self, mock_wizard):
        mock_wizard.model_type = ModelType.PYTORCH
        mock_wizard.model_path = "microsoft/phi-2"
        mock_wizard.mode = WizardMode.SMART
        mock_wizard.target_device = "cpu"
        mock_wizard.target_provider = "CPUExecutionProvider"
        mock_wizard.target_precision = "int4"

        config = mock_wizard.generate_config()

        assert config["input_model"]["type"] == "HfModel"
        assert config["input_model"]["model_path"] == "microsoft/phi-2"
        assert "passes" in config
```

### Integration Tests

```python
# tests/integ_tests/cli/test_init_integ.py

import subprocess
import json
import os
import pytest

class TestInitIntegration:
    def test_non_interactive_generates_config(self, tmp_path):
        result = subprocess.run([
            "olive", "init",
            "--model-type", "onnx",
            "--model", "test_model.onnx",
            "--smart",
            "--device", "cpu",
            "--provider", "CPUExecutionProvider",
            "--precision", "int8",
            "--output-dir", str(tmp_path),
            "--no-interactive"
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert os.path.exists(tmp_path / "olive-config.json")

        with open(tmp_path / "olive-config.json") as f:
            config = json.load(f)

        assert "input_model" in config
        assert "passes" in config
```

### E2E Tests

```python
# tests/e2e_tests/cli/test_init_e2e.py

import pexpect
import pytest

class TestInitE2E:
    @pytest.mark.slow
    def test_full_wizard_flow(self, tmp_path):
        """Test complete wizard interaction."""
        child = pexpect.spawn("olive init")

        # Model type
        child.expect("What type of model")
        child.sendline("")  # Select default (PyTorch)

        # Model source
        child.expect("How would you like to specify")
        child.sendline("")  # HuggingFace

        child.expect("Enter HuggingFace model name")
        child.sendline("microsoft/phi-2")

        # Mode
        child.expect("How would you like to configure")
        child.sendline("")  # Smart mode

        # Device
        child.expect("Select target device")
        child.sendline("")  # CPU

        # Provider
        # Precision
        child.expect("Select target precision")
        child.sendline("")  # INT4

        # Output
        child.expect("Output directory")
        child.sendline(str(tmp_path))

        # Action
        child.expect("What would you like to do")
        child.send("j")  # Move to "Generate configuration file"
        child.sendline("")

        child.expect("Configuration saved")
        child.wait()
```



## Appendix

### A. Pass Compatibility Matrix (Key Passes in `olive optimize`)

| Pass | CPU | CUDA | QNN | OpenVINO | VitisAI | DML | WebGPU | NvTensorRTRTX |
|------|-----|------|-----|----------|---------|-----|--------|---------------|
| ModelBuilder | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| OnnxConversion | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| OpenVINOOptimumConversion | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| QuaRot | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Gptq | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| OnnxStaticQuantization | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| OnnxBlockWiseRtnQuantization | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| OnnxFloatToFloat16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| OnnxPeepholeOptimizer | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| OrtTransformersOptimization | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| DynamicToFixedShape | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| StaticLLM | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| VitisAIAddMetaData | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| EPContextBinaryGenerator | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| OpenVINOIoUpdate | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| OpenVINOEncapsulation | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| OnnxIODataTypeConverter | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| SplitModel | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| ComposeOnnxModels | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |

### B. Quantization Algorithm Comparison

| Algorithm | Calibration | Speed | Quality | Best For |
|-----------|-------------|-------|---------|----------|
| RTN | Not needed | Fast | Good | Quick prototyping |
| GPTQ | Required | Slow | Excellent | Production LLMs |
| AWQ | Required | Medium | Excellent | LLMs with activations |
| HQQ | Not needed | Fast | Good | Memory-constrained ONNX models |
| QuaRot | Required | Slow | Excellent | QNN/Vitis deployment |
| SpinQuant | Required | Slow | Excellent | Rotation-based quantization |
| SeqMSE | Required | Slow | Excellent | AIMET sequential MSE |
| AdaRound | Required | Slow | Excellent | AIMET adaptive rounding |
| LPBQ | Required | Slow | Good | AIMET log-based binary quant |

### C. Command Reference

```bash
# Basic usage
olive init

# Skip to specific model type
olive init --model-type pytorch

# Non-interactive with all options
olive init \
    --model-type pytorch \
    --model "meta-llama/Llama-3.1-8B" \
    --smart \
    --device gpu \
    --provider CUDAExecutionProvider \
    --precision int4 \
    --output-dir ./output \
    --no-interactive

# Generate config only
olive init --output-dir ./config --generate-config-only
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-XX-XX | - | Initial design document |
| 1.1 | 2026-02-11 | - | Replaced User Flow section with detailed `olive init` Wizard Flow step-by-step design covering all model types (PyTorch/ONNX/Diffuser), branching logic, Auto/Custom modes, pass auto-selection tables, and generated command examples |
