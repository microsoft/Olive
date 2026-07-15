---
name: olive
description: Use Microsoft Olive through its native CLI and YAML/JSON workflow configuration files to optimize, export, quantize, fine-tune, tune, evaluate, and package AI models for ONNX Runtime. Use when a user mentions Olive, olive-ai, olive optimize, Olive passes, model conversion, ONNX optimization, execution providers, or asks to create, explain, validate, or run an Olive workflow config.
license: MIT
compatibility: Requires Python 3.10 or later and the olive-ai package. Model downloads and some dependency installations require network access; GPU, NPU, and vendor-specific workflows require matching hardware and runtimes.
metadata:
  author: microsoft
  version: "2.0.0"
---

# Microsoft Olive

Use the native `olive` command and Olive workflow files. This skill does not require or assume an MCP
server.

Treat the installed Olive version as the source of truth. Before using unfamiliar options, run:

```shell
olive --help
olive <command> --help
```

Do not invent command flags, pass names, pass parameters, model types, or execution providers. If an
existing project already has an Olive config, preserve its conventions and make the smallest necessary
change.

## Choose the right interface

| User goal | Preferred interface |
| --- | --- |
| Wants guided setup or does not know which operation to choose | `olive init` |
| Wants an end-to-end optimized model | `olive optimize` |
| Wants only ONNX export | `olive capture-onnx-graph` |
| Wants only quantization | `olive quantize` |
| Wants text LoRA or QLoRA training | `olive finetune` |
| Wants diffusion LoRA training | `olive diffusion-lora` |
| Wants one known Olive pass | `olive run-pass` |
| Wants a repeatable multi-pass pipeline, evaluation, search, or custom data | `olive run --config ...` |
| Wants ONNX Runtime session tuning | `olive tune-session-params` |
| Wants lm-eval benchmarking | `olive benchmark` |

Do not use `olive auto-opt`; it is deprecated in favor of `olive optimize`.

Read [the CLI guide](references/cli.md) for installation, command examples, dry runs, test mode, and
provider selection.

## Execution workflow

1. Identify the input model format and path or Hugging Face ID.
2. Identify the desired output: optimized ONNX, quantized model, adapter, benchmark, or reusable workflow.
3. Identify the target device and execution provider only when the user has not already specified them.
4. Check `olive <command> --help` in the active environment.
5. Use an explicit output directory and `--log_level 1` for meaningful progress logs.
6. For expensive or unfamiliar high-level commands, add `--dry_run`. Inspect the generated
   `<output_path>/config.json`, then run it with `olive run --config <path>` after it is correct.
7. Run the requested operation. Do not claim success until the process exits successfully and the expected
   output exists.
8. Report the output path, selected provider and precision, passes that ran, and metrics that Olive actually
   returned.

Use `--save_config_file` when the user wants both execution and a saved recipe. It saves
`olive_config.json` while the command continues. Use `--dry_run` when the user wants configuration
generation without optimization; it saves `config.json` and stops. It does not perform full workflow or
pass-schema validation.

## Write workflow configuration

Use a YAML or JSON workflow when the user needs multiple passes, reusable configuration, custom data,
evaluation, search, custom scripts, remote systems, or settings not exposed by a high-level command.

Read [the workflow configuration guide](references/workflow-config.md) before creating or editing a
workflow. Start from [the bundled workflow template](assets/workflow.yaml) or, preferably, generate a
version-specific config with a high-level command:

```shell
olive optimize \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --precision int4 \
  --provider CPUExecutionProvider \
  --output_path olive-output \
  --dry_run
```

When authoring a workflow:

- Keep `input_model`, `passes`, and output settings explicit.
- Define pass entries in execution order; mapping order is pipeline order.
- Reference systems, data configs, and evaluators by their declared names.
- Use a unique `name` for every data config.
- Use YAML for human-maintained files when comments are useful. JSON files must not contain comments or
  trailing commas.
- Never copy a model's Hugging Face `config.json` and treat it as an Olive workflow.
- Never place tokens, credentials, or secrets in a workflow file.

Before using a nontrivial pass, inspect its installed schema from this skill's root:

```shell
python scripts/inspect_pass.py OnnxConversion
```

Validate a workflow without running model optimization:

```shell
python scripts/validate_config.py workflow.yaml
olive run --config workflow.yaml --list_required_packages
```

The second command writes `olive_requirements.txt` in the current directory. Review it, install the listed
packages into the intended environment, and check model-loader and exporter requirements described in the
CLI guide before running:

```shell
olive run --config workflow.yaml
```

Structural validation cannot prove that remote models are accessible, local data is semantically correct,
the model is supported by every pass, or the target hardware has enough memory. Surface those constraints
instead of presenting validation as execution success.

## Dependency and hardware rules

- Reuse the user's active environment when it already contains the required Olive and runtime packages.
- Install dependencies only when Olive or a required optional package is missing.
- Use one ONNX Runtime variant per environment. Do not combine CPU, CUDA, DirectML, OpenVINO, or QNN
  runtime packages in the same environment unless the installed package documentation explicitly supports
  it.
- Match `device`, execution provider, and runtime: CPU with `CPUExecutionProvider`, NVIDIA GPU with
  `CUDAExecutionProvider`, Windows DirectX GPU with `DmlExecutionProvider`, and Qualcomm NPU with
  `QNNExecutionProvider`.
- Do not select fp16 for CPU merely to reduce model size. Prefer int4 or int8 when supported.
- Calibration-based quantization and fine-tuning may require datasets and substantial compute. Do not
  silently replace the user's dataset or algorithm.

## Safety and correctness

- Use `--trust_remote_code` only when the user explicitly trusts the model repository or existing project
  configuration already requires it.
- Use `HF_TOKEN` or the Hugging Face credential store for gated models. Never write a token into a command,
  config, script, log, or committed file.
- Do not enable `clean_cache`, delete outputs, overwrite a nonempty output directory, or remove generated
  artifacts without user intent.
- A `--test` run uses a small randomly initialized model with the same architecture. It checks pipeline
  compatibility, not real model quality.
- Do not claim quality, latency, memory, or size improvements without comparing actual outputs or metrics.
- If network access or hardware is unavailable, complete local config and dry-run validation and state
  exactly what remains unverified.
