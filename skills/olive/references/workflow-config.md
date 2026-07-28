# Olive Workflow Configuration

Olive workflows are YAML or JSON files consumed by:

```shell
olive run --config workflow.yaml
```

Olive accepts both YAML and JSON. Hand-maintained workflows may use YAML for comments, while
[microsoft/olive-recipes](https://github.com/microsoft/olive-recipes) normally uses JSON for executable
workflows. JSON does not allow comments or trailing commas.

## Choose a starting point

For a model- and provider-specific workflow, first find the same model or a close architecture in
`microsoft/olive-recipes`. Read the recipe README and use the JSON file named in its `olive run --config`
command. Files named `info.yml` or `info.yaml` describe the recipe for catalog and automation purposes; they
are not Olive workflow configs.

Reuse the closest recipe's model type, pass chain, system, provider, and data shape, then change only the
fields required for the user's model and output. Recipes can require backend-specific packages or target a
different Olive version, so validate the result against the active installation.

When no close recipe exists, generate a workflow for the installed Olive version:

```shell
olive optimize \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --precision int4 \
  --provider CPUExecutionProvider \
  --output_path generated-workflow \
  --dry_run
```

Olive writes `generated-workflow/config.json`. Edit that file rather than rebuilding a complex provider
recipe from memory.

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
