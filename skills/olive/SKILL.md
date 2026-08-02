---
name: olive
description: Use Microsoft Olive through its native CLI and YAML/JSON workflow configuration files to optimize, export, quantize, fine-tune, tune, evaluate, and package AI models for ONNX Runtime. Use when a user mentions Olive, olive-ai, olive optimize, Olive passes, model conversion, ONNX optimization, execution providers, or asks to create, explain, validate, or run an Olive workflow config.
license: MIT
compatibility: Requires Python 3.10 or later and the olive-ai package. Model downloads and some dependency installations require network access; GPU, NPU, and vendor-specific workflows require matching hardware and runtimes.
metadata:
  author: microsoft
  version: "2.3.0"
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

## Required target questions

Before searching recipes, generating a workflow, inspecting candidate pass chains, or running an Olive
command, confirm the target requirements with the user. Do not infer them from the model, provider defaults,
installed hardware, or the source checkpoint:

- Always confirm the requested output precision. For quantized workflows, confirm weight precision and
  activation precision separately when both apply.
- Confirm the execution provider if the user has not specified it.
- For QNN, confirm the target device or backend, such as QNN GPU versus HTP/NPU, and the exact Qualcomm SoC
  model or ORT `soc_model` value. If the user intentionally wants a portable non-AOT workflow, record that
  choice instead of inventing a SoC.
- For OpenVINO, confirm the target device, such as CPU, GPU, or NPU, and the target OpenVINO runtime or
  toolkit version.

Ask for every missing or ambiguous value before continuing. These values are recipe-search constraints:
include the model or architecture, provider, device, precision, and, when applicable, SoC model or runtime
version in the search. A recipe for the wrong device, SoC, precision, or runtime version is reference
material, not a drop-in starting point.

### Guide users who do not know the target details

Do not require the user to understand Olive precision names, execution-provider internals, QNN backends,
SoC IDs, or runtime versioning. If the user does not know a required value, switch to guided discovery
before searching for a final recipe:

1. Ask one plain-language question at a time. For precision, first ask whether the priority is quality,
   balanced quality and resource use, or minimum model size and highest feasible speed.
2. Determine whether the target is the current machine or another device. Inspect local hardware, available
   execution providers, SDKs, and runtime versions when the actual target is accessible; otherwise ask for
   the target product or chip name.
3. Inspect the source model configuration and existing weight format so unsupported or redundant precision
   conversions are not offered.
4. Map the user's goal and detected target to only the combinations supported by the installed Olive
   version, exporter, provider, and relevant model family.
5. Present a small set of valid choices with short tradeoffs, mark a recommended choice, and get explicit
   confirmation before searching recipes or generating a workflow.

For QNN, translate a product or chip name into QNN GPU versus HTP/NPU and the required SoC setting; do not
ask a non-expert for a numeric `soc_model` value when it can be derived from authoritative QNN or ONNX
Runtime documentation. For OpenVINO, detect the installed version and available target devices when
possible, then ask the user to confirm the intended device.

If a required target fact cannot be detected and the user cannot provide it, do not invent it. Explain which
decision remains unresolved. Produce a portable non-AOT or experimental scaffold only if the user explicitly
chooses that fallback, and do not label it as a hardware-specific final recipe.

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
3. Complete the required target questions above. Do not search recipes or generate a config while a required
   value is missing or ambiguous.
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
workflow. For a model- and provider-specific workflow, search
[microsoft/olive-recipes](https://github.com/microsoft/olive-recipes) before generating a generic config.
Read each selected recipe's README, executable workflow JSON, requirements, and version or commit pins;
`info.yml` and `info.yaml` are recipe catalog metadata, not files for `olive run --config`.

If the exact model and provider are absent, derive a candidate from prior recipes instead of stopping at an
exact-name search or merely replacing `model_path` in one recipe. Triangulate from:

- The same architecture or model family on any provider for model type, exporter, component layout, and
  architecture-specific graph transformations.
- The same provider and device for systems, lowering, static-shape, compilation or AOT passes, environment
  boundaries, and provider options.
- The same source weight format and quantization scheme for compatible quantization, calibration, and data
  settings.

Inspect the target model's Hugging Face configuration and repository metadata before merging those
references. Check its architecture, task, context and cache design, modality, parameter and checkpoint size,
and existing `quantization_config`. Do not schedule dequantization or a second quantizer for a pre-quantized
checkpoint unless the selected exporter explicitly supports that conversion.

Preserve multi-stage workflows from the provider recipe, such as separate quantization and QNN AOT
environments. Copy a pass or setting only when its model format, graph, precision, device, and runtime
preconditions still hold. Never infer that two models are compatible from repository names alone; a
distilled model can use a different base architecture than its name suggests.

Use a high-level dry run as an installed-version compatibility scaffold after studying the reference
recipes, not as the final inferred recipe:

```shell
olive optimize \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --precision int4 \
  --provider CPUExecutionProvider \
  --output_path olive-output \
  --dry_run
```

Compare its output with the reference pass chains, inspect every nontrivial pass schema, and explain which
recipe supplied each model-specific or provider-specific decision. If the installed exporter or a required
pass does not support the target architecture, report the recipe as blocked rather than presenting a
structurally valid config as runnable.

When authoring a workflow:

- Keep `input_model`, `passes`, and output settings explicit.
- Define pass entries in execution order; mapping order is pipeline order.
- Reference systems, data configs, and evaluators by their declared names.
- Use a unique `name` for every data config.
- Olive accepts JSON and YAML. Recipes normally use JSON for executable workflows; YAML remains useful for
  hand-maintained workflows with comments.
- Use [the bundled workflow template](assets/workflow.yaml) only as a generic Hugging Face-to-ONNX example.
  Current generative LLM recipes often use `ModelBuilder`; follow the closest recipe instead of substituting
  `OnnxConversion`.
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
  `CUDAExecutionProvider`, WebGPU with `WebGpuExecutionProvider`, Windows DirectX GPU with
  `DmlExecutionProvider`, and Qualcomm NPU with `QNNExecutionProvider`. Configure only one execution
  provider per accelerator.
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
