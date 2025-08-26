# Gemma-3-4B Model Optimization

This repository demonstrates the optimization of the [Google Gemma-3-4B](https://huggingface.co/google/gemma-3-4b-it) model using **post-training quantization (PTQ)** techniques for QNN (Qualcomm Neural Network) execution. The optimization process utilizes an environment based heavily upon the [PTQ tutorial for Phi-3.5](https://github.com/CodeLinaro/Olive/blob/main/examples/phi3_5/README.md)

## File Overview

This example contains the following key files:

- **`env_setup.sh`** - Automated environment setup script (Linux only)
- **`gemma3-4b-text-qnn-config.json`** - Olive configuration for optimizing the text component
- **`gemma3-4b-vision-qnn-config.json`** - Olive configuration for optimizing the vision component
- **`user_script.py`** - Dataset handling and preprocessing utilities
- **`custom_gemma3_4b_it_vision.py`** - Vision model loader for the optimization pipeline

## Prerequisites

### System Requirements
- **Operating System**: Linux (automated setup script is Linux-only)
- **Python**: 3.10
- **Package Manager**: [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)
- **Storage**: ~13GB for COCO train2017 dataset (downloaded automatically)

### Dependencies Installed by Setup Script
The `env_setup.sh` script installs the following components:
- setuptools (for building Olive from source)
- Olive requirements and dependencies
- AutoGPTQ (from source)
- GPTQModel (specific commit: `558449bed3ef2653c36041650d30da6bbbca440d`)
- onnxruntime-qnn (pre-release version)

## Setup Instructions

### Automated Setup (Recommended)
```bash
source env_setup.sh
```

### Manual Setup (Alternative)
If you prefer to set up manually or need to troubleshoot:

1. Install setuptools:
   ```bash
   uv pip install setuptools
   ```

2. Install requirements:
   ```bash
   uv pip install -r ../requirements.txt
   uv pip install -r ../../../requirements.txt
   ```

3. Install AutoGPTQ from source:
   ```bash
   export BUILD_CUDA_EXT=0
   uv pip install --no-build-isolation git+https://github.com/PanQiWei/AutoGPTQ.git
   ```

4. Install GPTQModel with Gemma3 fix:
   ```bash
   uv pip install --no-build-isolation git+https://github.com/ModelCloud/GPTQModel.git@558449bed3ef2653c36041650d30da6bbbca440d
   ```

5. Install onnxruntime-qnn:
   ```bash
   uv pip install -r https://raw.githubusercontent.com/microsoft/onnxruntime/refs/heads/main/requirements.txt
   uv pip install -U --pre --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple onnxruntime-qnn --no-deps
   ```

> **Important:** The setup uses a specific commit hash for GPTQModel (`558449bed3ef2653c36041650d30da6bbbca440d`) to address a [memory leak issue](https://github.com/ModelCloud/GPTQModel/commit/558449bed3ef2653c36041650d30da6bbbca440d) with Gemma3 models.

## Optimization Process

Since Gemma-3-4B is a multi-modal model composed of both vision and text components, the strategy for optimizing it through Olive is to operate on the constituent models separately before configuring them to work together at the onnxruntime-genai stage.

### Configuration Differences

**Text Configuration (`gemma3-4b-text-qnn-config.json`)**:
- Uses HuggingFace model directly (`google/gemma-3-4b-it`)
- Applies comprehensive optimization pipeline: QuaRot → GptqModel → ModelBuilder → Quantization
- Outputs to: `models/gemma-3-4b-it-text/`

**Vision Configuration (`gemma3-4b-vision-qnn-config.json`)**:
- Uses custom PyTorch model loader (`custom_gemma3_4b_it_vision.py`)
- Simpler pipeline: ONNX Conversion → Graph Surgery → Quantization
- Outputs to: `models/gemma-3-4b-it-vision/`

### Running Optimization

Execute the following commands to separately produce optimized binaries for each component:

```bash
olive run --config gemma3-4b-text-qnn-config.json
```

```bash
olive run --config gemma3-4b-vision-qnn-config.json
```

## Expected Outputs

After successful optimization, you will find:

- **Text model outputs**: `models/gemma-3-4b-it-text/`
- **Vision model outputs**: `models/gemma-3-4b-it-vision/`
- **Cache directory**: `cache/` (intermediate files and downloaded datasets)
- **Dataset**: `.cache/train2017/` (COCO train2017 images, ~13GB)

Both configurations use `"no_artifacts": true`, meaning only the final optimized models are retained.

## Troubleshooting

### Common Issues

**Insufficient Storage**: The COCO train2017 dataset requires ~13GB of storage and is downloaded automatically to `.cache/train2017/`.

**Memory Requirements**: The optimization process, particularly for the text model with its comprehensive pipeline, requires substantial memory.

**QNN Provider**: Ensure the QNNExecutionProvider is properly installed and configured in your environment.

**Platform Limitation**: The current setup script is designed for Linux only. Windows/macOS users will need to adapt the manual setup steps.

**Dataset Download**: If the COCO dataset download fails, check your internet connection and available storage. The script uses `wget` which must be available on your system.
