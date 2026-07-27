# Olive CLI Guide

The installed CLI is authoritative:

```shell
olive --help
olive <command> --help
```

Examples below use POSIX line continuation. Use backticks in PowerShell or place the command on one line.

## Installation

Create or reuse a virtual environment. Install exactly one ONNX Runtime provider extra:

| Target | Olive package | Common LLM Model Builder package |
| --- | --- | --- |
| CPU | `olive-ai[cpu]` | `onnxruntime-genai` |
| NVIDIA CUDA | `olive-ai[gpu]` | `onnxruntime-genai-cuda` |
| Windows DirectML | `olive-ai[directml]` | `onnxruntime-genai-directml` |
| OpenVINO | `olive-ai[openvino]` | Check the selected exporter |
| Qualcomm QNN | `olive-ai[qnn]` | Check the selected exporter |

Hugging Face models normally also need `transformers`. Combine extras when appropriate:

```shell
python -m pip install "olive-ai[cpu,capture-onnx-graph]" transformers onnxruntime-genai
```

Common operation extras:

| Operation | Extra |
| --- | --- |
| ONNX capture through Optimum | `olive-ai[capture-onnx-graph]` |
| LoRA or QLoRA fine-tuning | `olive-ai[finetune]` |
| Diffusion pipelines | `olive-ai[diffusers]` and `datasets` |
| OpenVINO passes | `olive-ai[openvino]` |
| Docker system | `olive-ai[docker]` |

Some quantizers and vendor passes have additional dependencies. For a workflow file, use
`olive run --config <file> --list_required_packages`. For a high-level command, inspect its help and the
error from the selected implementation rather than installing every optional package.

Do not install multiple ONNX Runtime variants into one environment. Use separate environments when
switching between CPU, CUDA, DirectML, OpenVINO, and QNN.

For Olive source development:

```shell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Guided setup

`olive init` interactively selects a model, operation, target, and output. It can generate a CLI command,
generate a reusable config, or run immediately:

```shell
olive init --output_path olive-output
```

Use it when requirements are vague or the user wants to explore supported choices.

## End-to-end optimization

`optimize` is the default all-in-one command. It schedules export, quantization, graph optimization, and
provider-specific passes:

```shell
olive optimize \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu \
  --provider CPUExecutionProvider \
  --precision int4 \
  --output_path models/qwen-int4 \
  --log_level 1
```

Useful options include:

- `--exporter model_builder|dynamo_exporter|torchscript_exporter|optimum_exporter`
- `--act_precision int8` for activation quantization where supported
- `--block_size <n>` for block-wise quantization
- `--surgeries <name...>` for graph surgeries
- `--num_split <n>` and `--memory <MB>` for model splitting
- `--enable_aot --qnn_env_path <path>` for supported QNN AOT workflows

Never use deprecated `olive auto-opt`.

## Generate a config before running

All high-level commands that expose `--dry_run` can generate a workflow from their command-line arguments
without executing passes:

```shell
olive optimize \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --precision int4 \
  --provider CPUExecutionProvider \
  --output_path recipes/qwen-int4 \
  --dry_run
```

This writes `recipes/qwen-int4/config.json`. Review or edit it, then execute:

```shell
olive run --config recipes/qwen-int4/config.json
```

Dry run checks command-line argument handling, but it does not perform full `RunConfig` or pass-schema
validation. Use `scripts/validate_config.py` and `olive run --list_required_packages` before execution.

`--save_config_file` is different: it saves `olive_config.json` and still executes the command.

## Export only

Use `capture-onnx-graph` when the user wants conversion without the full optimization schedule:

```shell
olive capture-onnx-graph \
  --model_name_or_path microsoft/Phi-4-mini-instruct \
  --use_model_builder \
  --precision int4 \
  --use_ort_genai \
  --output_path models/phi-onnx \
  --log_level 1
```

Exporter choices are mutually exclusive:

- `--use_model_builder` for supported generative models
- `--use_dynamo_exporter` for PyTorch Dynamo export
- `--use_mobius_builder` for supported multi-component or multimodal models; requires `mobius-ai`
- No exporter flag uses the command's default PyTorch/Optimum route

Use `--target_opset`, `--torch_dtype`, and `--fixed_param_dict` only when the exporter supports them.

## Quantize only

For a Hugging Face or PyTorch model:

```shell
olive quantize \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --algorithm rtn \
  --precision int4 \
  --implementation olive \
  --output_path models/qwen-rtn \
  --log_level 1
```

For an ONNX model using ONNX Runtime dynamic int8 quantization:

```shell
olive quantize \
  --model_name_or_path model.onnx \
  --algorithm rtn \
  --precision int8 \
  --implementation ort \
  --output_path models/model-int8 \
  --log_level 1
```

Supported combinations depend on input format, implementation, precision, QDQ encoding, and whether a
dataset is supplied. Run `olive quantize --help`; do not assume every algorithm works with every backend.
AWQ and GPTQ commonly need GPU resources and calibration data.

## Fine-tune

Text LoRA or QLoRA:

```shell
olive finetune \
  --model_name_or_path microsoft/Phi-4-mini-instruct \
  --method qlora \
  --data_name tatsu-lab/alpaca \
  --use_chat_template \
  --max_seq_len 1024 \
  --max_samples 256 \
  --output_path models/phi-adapter \
  --log_level 1
```

Use exactly one of `--text_field`, `--text_template`, or `--use_chat_template`. For local files, set
`--data_name` to the format (`json`, `csv`, or `parquet`) and use `--data_files`. Multiple CLI files are a
comma-separated list without spaces; split mappings require a workflow config.

Diffusion LoRA:

```shell
olive diffusion-lora \
  --model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --model_variant sdxl \
  --data_dir training-images \
  --max_train_steps 1000 \
  --output_path models/sdxl-adapter \
  --log_level 1
```

DreamBooth requires `--dreambooth` and `--instance_prompt`. Prior preservation additionally requires
`--with_prior_preservation` and `--class_prompt`.

## Run one pass

List pass names:

```shell
olive run-pass --list-passes
```

Run one pass with a JSON parameter object:

```shell
olive run-pass \
  --pass-name OnnxConversion \
  --model_name_or_path microsoft/Phi-4-mini-instruct \
  --pass-config '{"target_opset": 20}' \
  --output_path models/phi-onnx \
  --dry_run
```

Use `scripts/inspect_pass.py <PassName>` from the skill directory before constructing nontrivial
`--pass-config` JSON.

## Run a workflow file

Olive accepts `.json`, `.yaml`, and `.yml`:

```shell
python scripts/validate_config.py workflow.yaml
olive run --config workflow.yaml --list_required_packages
python -m pip install -r olive_requirements.txt
olive run --config workflow.yaml --log_level 1
```

The `run` command can override selected model, output, and logging fields:

```shell
olive run \
  --config workflow.yaml \
  --model_name_or_path another/model \
  --output_path another-output \
  --log_level 1
```

Only use overrides when intentional; otherwise the workflow file should remain the reproducible source of
truth.

`--list_required_packages` reports dependencies declared by passes and the selected runtime. It may not
include every model-loader or exporter dependency. For example, a Hugging Face `OnnxConversion` workflow
normally also needs `transformers` and either `optimum` or a separately configured exporter.

## Tune and benchmark

Tune ONNX Runtime session settings:

```shell
olive tune-session-params \
  --model_name_or_path model.onnx \
  --device cpu \
  --providers_list CPUExecutionProvider \
  --output_path tuned-session \
  --log_level 1
```

Evaluate with lm-eval:

```shell
olive benchmark \
  --model_name_or_path model-or-hf-id \
  --tasks hellaswag \
  --device cpu \
  --limit 0.1 \
  --output_path benchmark-output \
  --log_level 1
```

`ort` and `ortgenai` benchmark backends require ONNX input; `ortgenai` also requires generated
`genai_config.json` assets.

## Pipeline test mode

For Hugging Face inputs, `--test` replaces downloaded weights with a small randomly initialized model of
the same architecture:

```shell
olive optimize \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --precision int4 \
  --output_path test-output \
  --test \
  --test_metrics mae speedup
```

Use a dedicated output directory. Test mode validates architecture and pipeline compatibility; it does not
measure real model quality.

## Authentication and remote code

Authenticate gated models through the Hugging Face credential store or an environment variable:

```shell
huggingface-cli login
```

Never put a token in an Olive config or a committed command script. Add `--trust_remote_code` only after the
user has explicitly accepted the model repository's code.
