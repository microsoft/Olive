# Olive Workflow Configuration

Olive workflows are YAML or JSON files consumed by:

```shell
olive run --config workflow.yaml
```

Olive accepts both YAML and JSON. Hand-maintained workflows may use YAML for comments, while
[microsoft/olive-recipes](https://github.com/microsoft/olive-recipes) normally uses JSON for executable
workflows. JSON does not allow comments or trailing commas.

## Collect search constraints first

Do not search for or synthesize a recipe until the target constraints are known:

- Precision: always confirm the requested output precision. For quantized workflows, distinguish weight
  precision from activation precision.
- Provider: confirm the intended execution provider.
- QNN: confirm QNN GPU versus HTP/NPU, the exact Qualcomm SoC model or ORT `soc_model` value, and whether
  the workflow must produce an AOT context binary.
- OpenVINO: confirm the CPU, GPU, or NPU target and the exact OpenVINO runtime or toolkit version.

Ask the user for missing values rather than selecting defaults. Use the resulting model or architecture,
provider, device, precision, SoC, runtime version, and output form as recipe-search terms. Provider-only
searches are insufficient because QNN GPU and HTP/NPU workflows differ, as can OpenVINO workflows across
devices and releases.

## Choose and synthesize a starting point

For a model- and provider-specific workflow, search `microsoft/olive-recipes` before generating a generic
workflow. An exact model-and-provider recipe is the strongest starting point. Read its README and use the
JSON file named in its `olive run --config` command. Files named `info.yml` or `info.yaml` describe the
recipe for catalog and automation purposes; they are not Olive workflow configs.

An exact-name miss does not mean there is no useful prior art. Build a reference set from the closest
available recipes:

- Same architecture or base model on any provider: learn the input model type, exporter, component layout,
  cache I/O, and graph surgeries.
- Same provider and device with the closest compatible architecture: learn the systems, execution provider,
  static shapes, compilation or AOT stages, and provider options.
- Same source weight format or quantization: learn compatible precision conversions, quantizers,
  calibration data, and block or group settings.
- Similar model scale or component count: learn splitting strategies, memory constraints, and staged
  outputs.

Repository names are not architecture evidence. For example, a distilled model can use Qwen or Llama
internals while a newer model from the same publisher can introduce an unsupported architecture. Confirm
the target model's `architectures`, `model_type`, task, context/cache design, modality, parameter count,
checkpoint size, and `quantization_config` from its model configuration and repository metadata.

Read the README, all executable configs used by its commands, requirements, and Olive/runtime version pins
for every selected reference. Provider recipes may intentionally separate quantization, export, and AOT
compilation into different configs and Python environments. Preserve those stage boundaries unless the
installed pass documentation explicitly supports combining them.

Synthesize the candidate by assigning each concern to the most relevant reference:

1. Start with the exporter and graph structure from the architecture reference.
2. Apply systems, provider lowering, static-shape, and compilation stages from the provider reference.
3. Apply quantization only from a reference with a compatible input weight format and exporter path.
4. Carry over data shapes and calibration only when the target model and pass require the same semantics.
5. Keep pass ordering, intermediate model types, and environment handoffs intact.
6. Record which recipe supports each inherited pass or non-obvious setting.

Architecture-specific graph surgeries may be required. Retain or adapt them when the target uses the same
architecture and exporter and the exported graph satisfies the surgery's pattern and pass preconditions.
Do not transfer surgeries, cache names, tensor shapes, or quantization settings solely because they appear
in the closest provider recipe. Check each pass's input model type, output model type, architecture
assumptions, precision support, device, and execution provider. If the source checkpoint is already FP8,
FP4, GPTQ, AWQ, or another quantized format, do not blindly add a second quantizer; first verify that the
exporter can consume or intentionally convert that representation.

For QNN, distinguish QNN GPU from HTP/NPU targets by the recipe's accelerator and provider options rather
than assuming that `QNNExecutionProvider` always means NPU. Preserve a recipe's separate host quantization
and QNN compilation environments, including its `PythonEnvironment`, intermediate model path, and context
binary stage.

After deriving the candidate, generate a workflow for the installed Olive version as a compatibility
scaffold:

```shell
olive optimize \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --precision int4 \
  --provider CPUExecutionProvider \
  --output_path generated-workflow \
  --dry_run
```

Olive writes `generated-workflow/config.json`. Compare it with the reference set instead of accepting it as
the final recipe. Keep installed-version field names and defaults where they are compatible, then restore
recipe-specific stages that the high-level command cannot express. A dry run validates argument handling;
it is not evidence that the exporter, passes, model architecture, or target runtime support the model.

When no adequate architecture or provider reference exists, use the generated workflow only as a clearly
labeled experimental scaffold. Do not present it as a runnable inferred recipe until the required exporter
and pass compatibility has been established.

For a hand-authored starting point, copy `assets/workflow.yaml` from this skill. It is a classic Hugging
Face-to-ONNX conversion and graph-optimization example, not a universal template for current generative
LLMs. It normally needs `olive-ai[cpu,optimum]` and `transformers`.

## Top-level structure

```yaml
workflow_id: my_workflow
input_model: {}
systems: {}
data_configs: []
evaluators: {}
passes: {}
search_strategy: false
host: null
target: null
evaluator: null
output_dir: olive-output
cache_dir: .olive-cache
log_severity_level: 1
```

`input_model` is required. Other sections are optional. Engine settings can be written under an `engine`
object or flattened at the top level; prefer the flattened form emitted by current high-level CLI commands.

## Input models

### Hugging Face model

```yaml
input_model:
  type: HfModel
  model_path: Qwen/Qwen2.5-0.5B-Instruct
  task: text-generation-with-past
  adapter_path: null
  load_kwargs:
    trust_remote_code: false
    attn_implementation: eager
```

`model_path` and `adapter_path` may be local paths or Hub IDs. Avoid `trust_remote_code: true` unless the
user explicitly trusts the repository.

### Local PyTorch model

```yaml
input_model:
  type: PyTorchModel
  model_path: model.pt
  model_script: model_loader.py
  model_loader: load_model
  io_config:
    input_names: [input_ids, attention_mask]
    output_names: [logits]
    input_shapes:
      - [1, 128]
      - [1, 128]
  dummy_inputs_func: create_dummy_inputs
```

`model_loader`, `io_config`, and `dummy_inputs_func` can refer to functions in `model_script`. Supply
`script_dir` when the script imports local modules from another directory.

### ONNX model

Single file:

```yaml
input_model:
  type: ONNXModel
  model_path: models/model.onnx
```

Model with external data:

```yaml
input_model:
  type: ONNXModel
  model_path: models/model
  onnx_file_name: model.onnx
```

For external data, `model_path` is the containing directory and `onnx_file_name` identifies the graph.

## Systems and execution providers

Use explicit systems when the target provider matters:

```yaml
systems:
  local_cpu:
    type: LocalSystem
    accelerators:
      - device: cpu
        execution_providers:
          - CPUExecutionProvider
host: local_cpu
target: local_cpu
```

Common mappings:

| Device | Execution provider |
| --- | --- |
| CPU | `CPUExecutionProvider` |
| NVIDIA GPU | `CUDAExecutionProvider` |
| WebGPU | `WebGpuExecutionProvider` |
| Windows DirectX GPU or NPU | `DmlExecutionProvider` |
| Intel OpenVINO | `OpenVINOExecutionProvider` |
| Qualcomm NPU | `QNNExecutionProvider` |

Only one accelerator is currently supported per system, and each accelerator accepts one execution
provider. If systems are omitted, Olive defaults to the local system and infers available providers where
possible. Do not copy older recipes that list multiple providers under one accelerator.

`host` is where passes execute. `target` is where evaluation and target-side passes execute. They may
refer to the same system or different systems.

## Passes

Pass keys are unique user-chosen labels. `type` is the registered Olive pass class. Entries execute in
mapping order:

```yaml
passes:
  conversion:
    type: OnnxConversion
    target_opset: 20
    save_as_external_data: true
    all_tensors_to_one_file: true
  graph_optimization:
    type: OrtTransformersOptimization
  session_tuning:
    type: OrtSessionParamsTuning
    data_config: token_data
    io_bind: true
```

Pass parameters may be flattened beside `type`, as above. `host` and `evaluator` can override the
workflow-level values for one pass. A nested `config` object is also accepted, but current recipes generally
use flattened pass parameters. Follow the selected recipe and installed pass schema rather than moving
fields between the two forms blindly.

Many current generative LLM recipes use `ModelBuilder` rather than the traditional `OnnxConversion` pass:

```yaml
passes:
  model_builder:
    type: ModelBuilder
    precision: int4
```

Some backends add `GraphSurgeries` or backend-specific passes. Copy the complete chain from the closest
model/provider recipe; do not infer it from this minimal example.

Inspect available passes and exact parameters in the active Olive installation:

```shell
olive run-pass --list-passes
python scripts/inspect_pass.py OnnxConversion
python scripts/inspect_pass.py OnnxBlockWiseRtnQuantization \
  --device cpu \
  --provider CPUExecutionProvider
```

Do not use a pass merely because its name is plausible. Check its supported model format, accelerator,
provider, precision, algorithm, dataset requirement, optional packages, and parameter schema.

## Data configs

Data configs are declared once and referenced by name from passes and metrics:

```yaml
data_configs:
  - name: calibration_data
    type: HuggingfaceContainer
    load_dataset_config:
      data_name: Salesforce/wikitext
      subset: wikitext-2-raw-v1
      split: train
    pre_process_data_config:
      strategy: line-by-line
      max_samples: 128
      max_seq_len: 512
    dataloader_config:
      batch_size: 1
```

Every data config name must be unique and contain letters, numbers, and underscores.

Common container types:

- `HuggingfaceContainer` for Hub datasets and local CSV, JSON, JSONL, or Parquet files
- `DummyDataContainer` for explicit input names, shapes, and dtypes
- `TransformersPromptDummyDataContainer` for prompt-phase transformer inputs
- `TransformersTokenDummyDataContainer` for token-phase transformer inputs with KV cache
- `RawDataContainer` for raw tensor files

Local JSONL example:

```yaml
data_configs:
  - name: local_training_data
    type: HuggingfaceContainer
    load_dataset_config:
      data_name: json
      data_files:
        train: data/train.jsonl
        validation: data/validation.jsonl
      split: train
    pre_process_data_config:
      type: text_generation_huggingface_pre_process
      text_cols: text
    dataloader_config:
      batch_size: 1
```

File-to-split mappings are supported in workflow files even though the high-level CLI accepts only one
file or a comma-separated file list.

Reference a data config from a pass:

```yaml
passes:
  static_quantization:
    type: OnnxStaticQuantization
    data_config: calibration_data
```

Inline data configs in pass or metric fields are not supported; use the declared name.

## Evaluators and metrics

Evaluators contain metrics and can be attached to the engine or an individual pass:

```yaml
evaluators:
  common_evaluator:
    metrics:
      - name: accuracy
        type: accuracy
        data_config: evaluation_data
        sub_types:
          - name: accuracy_score
            priority: 1
            goal:
              type: max-degradation
              value: 0.01
      - name: latency
        type: latency
        data_config: latency_data
        sub_types:
          - name: avg
            priority: 2
            goal:
              type: percent-min-improvement
              value: 20
evaluator: common_evaluator
```

Built-in metric types include `accuracy`, `latency`, `throughput`, `size_on_disk`, and `custom`. Multiple
objectives need distinct priorities so Olive can rank candidates. Set `higher_is_better` explicitly for a
custom or unusual metric.

An evaluator is required for parameter search. Without an evaluator, omit search and use fixed pass
parameters.

Current language-model recipes may use `LMEvaluator` directly instead of a `metrics` list:

```yaml
evaluators:
  mmlu:
    type: LMEvaluator
    tasks: [mmlu]
    batch_size: 8
evaluator: mmlu
```

## Search

Disable search for a deterministic ordered pipeline:

```yaml
search_strategy: false
```

Enable search with an explicit strategy:

```yaml
search_strategy:
  execution_order: joint
  sampler: tpe
  max_samples: 5
  seed: 0
  stop_when_goals_met: true
evaluator: common_evaluator
```

Do not use `search_strategy: true`; current Olive versions require an explicit strategy mapping. Only enable
search after defining metrics, priorities, and goals. A config with search enabled but no evaluator is
invalid.

Pass parameters can use fixed values, `DEFAULT_VALUE`, `SEARCHABLE_VALUES`, or pass-supported lists that
form a categorical search space. Inspect the pass schema before using searchable values.

## Output, cache, logging, and packaging

```yaml
output_dir: outputs/my-workflow
cache_dir: .olive-cache
clean_cache: false
clean_evaluation_cache: false
evaluate_input_model: false
log_severity_level: 1
ort_log_severity_level: 3
ort_py_log_severity_level: 3
log_to_file: false
packaging_config:
  type: Zipfile
  name: optimized-model
```

Logging levels are `0` debug, `1` info, `2` warning, `3` error, and `4` critical. Keep `clean_cache` false
unless the user intends to discard cached intermediate results. `no_artifacts: true` suppresses auxiliary
metrics, footprints, and run-history artifacts; it does not prevent the final model from being written to
`output_dir`.

## Custom code

Custom script fields belong to the object that consumes them; `user_script` and `script_dir` are not
top-level `RunConfig` fields.

| Purpose | Configuration scope |
| --- | --- |
| PyTorch model loader and model helpers | `input_model.model_script` and `input_model.script_dir` |
| Registered dataset or data-processing components | `data_configs[*].user_script` and `script_dir` |
| Pass-specific helper code | The pass's `user_script` and `script_dir` fields, only when its schema exposes them |
| Custom metric code | The metric's `user_config.user_script` and `user_config.script_dir` |
| Custom evaluator implementation | `evaluators.<name>.user_script` and `evaluators.<name>.script_dir` |

Use registered function names in the corresponding model, data, pass, metric, or evaluator fields. Do not
embed arbitrary Python code in YAML or JSON.

## Validation and execution

From the skill root, validate structure, references, pass names, pass parameters, and declared local
packages:

```shell
python scripts/validate_config.py workflow.yaml
```

Ask Olive to generate a dependency file without running passes:

```shell
olive run --config workflow.yaml --list_required_packages
```

This writes `olive_requirements.txt` in the current directory. Review and install it:

```shell
python -m pip install -r olive_requirements.txt
```

This file contains dependencies declared by passes and the selected runtime. Check model-loader and
exporter requirements separately; Hugging Face conversion commonly also requires `transformers` and
`optimum`, while Model Builder workflows require the matching `onnxruntime-genai` package.

Then execute:

```shell
olive run --config workflow.yaml --log_level 1
```

Validation does not download the model or dataset and cannot guarantee runtime memory, hardware support,
or model/pass semantic compatibility.
