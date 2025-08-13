# Microsoft Olive - AI Model Optimization Toolkit

Microsoft Olive is a Python-based AI model optimization toolkit for the ONNX Runtime. It provides 40+ built-in optimization components for model compression, optimization, fine-tuning, and compilation.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap the Development Environment
- Install basic dependencies: `pip install -r requirements.txt` -- takes 3 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
- Install Olive in development mode: `pip install -e .` -- takes 10 seconds.
- Install development dependencies: `pip install -r requirements-dev.txt` -- takes 20 seconds.
- Initialize linting: `lintrunner init` -- takes 15 seconds.

### Build and Test Commands
- Lint code: `lintrunner` -- takes 1-2 seconds for single files, 5+ seconds for entire codebase. NEVER CANCEL.
- Lint with auto-fix: `lintrunner -a`
- Import test: `python -c "import olive; print('Olive version:', olive.__version__)"`

### Command Line Interface
- Main CLI entry point: `olive --help`
- Key commands:
  - `olive optimize` -- Comprehensive model optimization with pass scheduling
  - `olive auto-opt` -- Automatic optimization with minimal configuration
  - `olive finetune` -- Fine-tune models using PEFT
  - `olive quantize` -- Quantize models
  - `olive capture-onnx-graph` -- Export models to ONNX format
- Test CLI functionality: `olive optimize --help` and `olive auto-opt --help`
- Dry run mode: `olive optimize --dry_run` (validates configuration without execution)

### Network Limitations
- **CRITICAL**: The environment has limited network connectivity
- HuggingFace model downloads may fail with connection errors
- PyPI package installations may timeout - use longer timeouts (10-20 minutes) for ML packages
- When network issues occur, focus on local functionality and offline validation

## Validation

### Pre-commit Validation
- ALWAYS run `lintrunner` before committing changes or the CI (.github/workflows/lint.yml) will fail
- Test basic functionality: `python -c "import olive; print(olive.__version__)"`
- Validate CLI: `olive --help`

### Manual Testing Scenarios
- Test package import and version: `python -c "import olive; print('Version:', olive.__version__)"`
- Validate CLI help commands work: `olive --help`, `olive optimize --help`, `olive auto-opt --help`
- Test dry run mode: `olive optimize --dry_run --model_name_or_path <model> --precision int4`
- Note: Full model optimization requires network access for HuggingFace downloads

### Expected Timing and Timeout Values
- **NEVER CANCEL**: Basic requirements install: 3 minutes (set timeout to 10+ minutes)
- Development install: 10 seconds (set timeout to 60 seconds)
- Dev dependencies: 20 seconds (set timeout to 300 seconds)
- Lintrunner init: 15 seconds (set timeout to 600 seconds)
- Linting single file: 1-2 seconds
- Linting full codebase: 5-10 seconds (set timeout to 300 seconds)
- Test dependencies install: **FAILS due to network timeouts** - expect 20+ minute installs in normal environments

## Common Tasks

### Repository Structure
- `olive/` -- Main package directory with core functionality
- `test/` -- Test suite with requirements-test*.txt files
- `examples/` -- Example scripts and notebooks (35+ model optimization examples)
- `docs/` -- Documentation
- `.github/workflows/` -- CI/CD pipelines (lint.yml, codeql.yml)

### Key Configuration Files
- `pyproject.toml` -- Modern Python packaging configuration with ruff, pylint, pytest settings
- `setup.py` -- Package setup and console_scripts entry point
- `requirements*.txt` -- Dependency specifications
- `.lintrunner.toml` -- Linting configuration
- `olive_config.json` -- Olive package extras configuration

### Python Package Information
- Package name: `olive-ai`
- Python version: 3.9+ (currently testing on 3.12)
- Entry point: `olive` command (defined in setup.py console_scripts)
- Version location: `olive/__init__.py` (__version__)

### Examples and Documentation
- Getting started notebooks: `examples/getting_started/`
- Model-specific examples: `examples/{model_name}/` (phi2, mistral, llama, etc.)
- Quick start example: Qwen/Qwen2.5-0.5B-Instruct optimization
- Documentation website: https://microsoft.github.io/Olive/

### Development Workflow
- Code style: Black formatter with 120 character line limit
- Linting: Multiple tools via lintrunner (ruff, pylint, editorconfig-checker, etc.)
- Testing: pytest framework (note: full test suite requires network access)
- Pre-commit hooks: Available via .pre-commit-config.yaml

### Known Issues and Workarounds
- Network connectivity may prevent HuggingFace model downloads
- Test dependency installation may timeout due to network issues
- Use `--dry_run` mode for configuration validation without network access
- Focus on local testing and validation when network is unreliable

## Working Offline
When network connectivity is limited:
- Use local model paths instead of HuggingFace model names
- Focus on configuration validation with `--dry_run` mode
- Test CLI help commands and basic imports
- Validate linting and code formatting
- Use local examples in `examples/` directory for reference