# AGENTS.md

## Project overview

Olive is a Python toolkit for AI model optimization on ONNX Runtime. The main package is in `olive/`, tests are in `test/`, and the CLI entry point is `olive`.

## User guide

Olive users should install `olive-ai` with `pip install olive-ai`, and follow `README.md`.

## Developer setup and validation

1. `pip install -r requirements.txt`
2. `pip install -e .`
3. `pip install -r requirements-dev.txt`
4. `lintrunner init`

Use the smallest relevant check for your change:

- `python -c "import olive; print(olive.__version__)"`
- `olive --help`
- `lintrunner --all-files -a --skip PYLINT` for faster local linting while iterating; run PYLINT at the end when changes are ready for a PR

If you need to run tests, install the appropriate dependencies from `test/requirements-test.txt` or one of the platform-specific files in `test/`.

## Repository map

- `olive/`: main package
  - `cli/`: CLI commands such as `optimize`, `quantize`, `finetune`, `capture-onnx-graph`, plus the shared launcher and base CLI logic.
  - `common/`: shared helpers for config loading, Hugging Face integration, quantization helpers, ONNX/ORT utilities, and user module loading.
  - `data/`: data configuration and registry.
    - `component/`: dataset, dataloader, dataset loading, and pre/post-processing components.
    - `container/`: data container implementations such as Hugging Face, raw, image, and dummy containers.
  - `engine/`: workflow execution, run footprint tracking, outputs, and packaging helpers.
  - `evaluator/`: metrics, metric backends, metric config/result types, and evaluator orchestration.
  - `hardware/`: accelerator definitions and related constants.
  - `model/`: model definitions and runtime-specific model handling.
    - `config/`: model config, Hugging Face config, I/O config, KV-cache config, and model registry helpers.
    - `handler/`: handlers for composite, diffusers, Hugging Face, ONNX, OpenVINO, PyTorch, QAIRT, and QNN models.
    - `utils/`: model utility helpers.
  - `passes/`: optimization, conversion, and quantization passes.
    - Backend-specific pass groups include `onnx/`, `pytorch/`, `openvino/`, `qnn/`, `qairt/`, `diffusers/`, and `quark_quantizer/`.
  - `platform_sdk/qualcomm/`: Qualcomm SDK environment setup, runners, and helper scripts.
  - `search/`: search space, search points/results/samples, strategies, and sampler implementations.
  - `systems/`: execution targets and runners for local, Docker, and Python-environment based systems.
  - `telemetry/`: telemetry plumbing, exporters, utilities, and device ID support.
  - `workflows/run/`: workflow config and workflow run entrypoints.
- `test/`: test suite, organized to largely mirror the package structure.
  - Main areas include `cli/`, `common/`, `data_container/`, `engine/`, `evaluator/`, `hardware/`, `model/`, `passes/`, `resource_path/`, `search/`, `systems/`, and `workflows/`.
  - `assets/`: shared fixtures and helper scripts for tests.
  - `requirements-test*.txt`: base and platform-specific test dependency sets.
- `docs/`: documentation source and build config.
  - `source/` contains `getting-started`, `how-to`, `reference`, `features`, `blogs`, images, and Sphinx config.
- `notebooks/`: example notebooks for quickstart, finetuning, LoRA, and optimized text generation workflows.
- `scripts/`: repository maintenance and artifact-generation utilities.
- `.github/workflows/`: GitHub Actions workflows, including the Python lint workflow.

## Code style

- Follow Black-compatible formatting with a 120 character line length.
- Prefer existing repository patterns before adding new abstractions.
- Use absolute imports.
- Use `pathlib` for path handling.
- Keep changes focused and avoid unrelated refactors.

## Testing

- Add or update tests for behavior changes.
- Prefer behavior-focused tests over implementation-focused tests.
- Use the existing naming convention: `test_<method_or_function>_<expected_behavior>[_when_<condition>]`.

## Security and environment notes

- Do not commit secrets, caches, or generated model artifacts.
- Some workflows require optional packages, specific hardware, or model downloads. In limited-network environments, prefer local validation and `olive optimize --dry_run ...`.
