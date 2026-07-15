# Olive MCP Tool Reference

Use this reference for exact tool defaults and constraints. Omit optional arguments unless the user needs
them; the MCP server filters out `None` values.

## Supported values

Providers:

`CPUExecutionProvider`, `CUDAExecutionProvider`, `DmlExecutionProvider`, `OpenVINOExecutionProvider`,
`TensorrtExecutionProvider`, `ROCMExecutionProvider`, `QNNExecutionProvider`,
`VitisAIExecutionProvider`, `WebGpuExecutionProvider`, `NvTensorRTRTXExecutionProvider`

Precisions:

`fp32`, `fp16`, `bf16`, `int4`, `int8`, `int16`, `int32`, `uint4`, `uint8`, `uint16`, `uint32`

Quantization algorithms:

`rtn`, `gptq`, `awq`, `hqq`

The MCP server rejects fp16 and bf16 with `CPUExecutionProvider`.

## Planning and lifecycle tools

### `detect_hardware()`

Returns CPU architecture and core count, total and available RAM, NVIDIA GPU and VRAM information from
`nvidia-smi`, free disk space, OS information, and recommendations. It does not detect AMD or Intel GPUs,
although Windows systems may still support DirectML.

### `get_job_status(job_id, last_n_logs=50)`

Long-polls for up to 30 seconds and returns:

- `status`, `command`, `description`, and elapsed time
- `new_lines`, `recent_logs`, total log count, and seconds since the last output
- Active phase information and the current pass when detectable
- Final `result` after completion, failure, or cancellation

Finished successful or failed jobs are retained in server memory for one hour. A server restart loses
in-memory job status but does not delete output files.

### `cancel_job(job_id)`

Terminates a job in `starting`, `setting_up`, or `running` state. Do not call it without user intent.

## Model operations

All model operations return immediately with a `job_id` after validation and environment resolution.
The server runs each job in an isolated environment with operation-specific dependencies.

### `optimize`

End-to-end optimization with automatic pass scheduling.

| Parameter | Default | Notes |
| --- | --- | --- |
| `model_name_or_path` | Required | Hugging Face model ID or local path |
| `provider` | `CPUExecutionProvider` | Target ONNX Runtime execution provider |
| `device` | Auto | `cpu`, `gpu`, or `npu` |
| `precision` | `fp32` | Target precision |
| `act_precision` | None | Optional activation precision |
| `exporter` | Auto | `model_builder`, `dynamo_exporter`, `torchscript_exporter`, or `optimum_exporter` |
| `use_qdq_format` | `false` | Use QDQ instead of QOperator format |
| `num_split` | None | Number of model splits |
| `memory` | None | Available device memory in MB |
| `block_size` | None | Quantization block size; `-1` means per-channel |
| `surgeries` | None | List of graph surgeries |
| `output_path` | Auto | Timestamped output directory |
| `hf_token` | None | User-provided token for gated or private models |

Int4 or uint4 optimization enables calibration data dependencies. On CPU, RTN through `quantize` is
usually much faster.

### `quantize`

Focused model quantization.

| Parameter | Default | Notes |
| --- | --- | --- |
| `model_name_or_path` | Required | Hugging Face model ID or local path |
| `algorithm` | `rtn` | `rtn`, `gptq`, `awq`, or `hqq` |
| `precision` | `int8` | Target precision |
| `act_precision` | None | Activation precision for static quantization |
| `implementation` | `olive` | Backend such as `olive`, `ort`, `bnb`, `nvmo`, or `inc` |
| `use_qdq_encoding` | `false` | Encode quantized ONNX nodes with QDQ |
| `data_name` | None | Hugging Face calibration dataset when required |
| `output_path` | Auto | Timestamped output directory |
| `hf_token` | None | User-provided token for gated or private models |

Calibration-based algorithms may require `data_name`. Do not invent a dataset when the user has specified
one.

### `finetune`

Routes text models to LoRA or QLoRA and recognizable Stable Diffusion, SDXL, Flux, or diffusion models to
diffusion LoRA training.

Shared parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `model_name_or_path` | Required | Hugging Face model ID or local path |
| `data_name` | None | Required for text; one of `data_name` or `data_dir` is required for diffusion |
| `lora_r` | `64` | LoRA rank |
| `output_path` | Auto | Timestamped adapter directory |
| `hf_token` | None | User-provided token for gated or private models |

Text-only parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `method` | `lora` | `lora` or lower-memory `qlora` |
| `lora_alpha` | `16` | LoRA scaling |
| `target_modules` | None | Comma-separated module names |
| `torch_dtype` | `bfloat16` | `bfloat16`, `float16`, or `float32` |
| `train_split` | `train` | Training dataset split |
| `eval_split` | None | Optional evaluation split |

Diffusion-only parameters:

| Parameter | Default | Notes |
| --- | --- | --- |
| `data_dir` | None | Local image directory |
| `model_variant` | `auto` | `auto`, `sd`, `sdxl`, or `flux` |
| `alpha` | Same as rank | Diffusion LoRA scaling |
| `max_train_steps` | `1000` | Maximum training steps |
| `learning_rate` | `1e-4` | Learning rate |
| `train_batch_size` | `1` | Training batch size |
| `mixed_precision` | `bf16` | `bf16`, `fp16`, or `no` |
| `dreambooth` | `false` | Enable DreamBooth |
| `instance_prompt` | None | Required by the selected DreamBooth workflow when applicable |
| `merge_lora` | `false` | Merge the adapter into the base model |

### `capture_onnx_graph`

Captures an ONNX graph from a Hugging Face or local model.

| Parameter | Default | Notes |
| --- | --- | --- |
| `model_name_or_path` | Required | Hugging Face model ID or local path |
| `use_model_builder` | `false` | Use ONNX Runtime Model Builder |
| `use_dynamo_exporter` | `false` | Use the PyTorch dynamo exporter |
| `precision` | `fp16` | Model Builder precision: `fp16`, `fp32`, `int4`, or `bf16` |
| `conversion_device` | `cpu` | `cpu` or `gpu` |
| `torch_dtype` | None | Optional source-model cast such as `float16` |
| `target_opset` | `20` | Target ONNX opset |
| `use_ort_genai` | `false` | Use the ONNX Runtime generate API |
| `output_path` | Auto | Timestamped output directory |
| `hf_token` | None | User-provided token for gated or private models |

### `benchmark`

Evaluates a model with lm-eval tasks.

| Parameter | Default | Notes |
| --- | --- | --- |
| `model_name_or_path` | Required | Hugging Face model ID or local path |
| `tasks` | `["hellaswag"]` | List of lm-eval tasks |
| `device` | `cpu` | `cpu` or `gpu` |
| `batch_size` | `1` | Evaluation batch size |
| `max_length` | `1024` | Maximum combined input and output length |
| `limit` | `1.0` | Dataset fraction from 0 to 1, or an absolute sample count |
| `output_path` | Auto | Timestamped output directory |
| `hf_token` | None | User-provided token for gated or private models |

## Output management

### `manage_outputs(action="list", prefix=None, names=None, delete_all=false, limit=20)`

- `action="list"` returns recent output directories, inferred operation and timestamp, up to five discovered
  model files, and whether a known config file exists.
- `action="delete"` deletes exact directory `names`, every directory matching an operation `prefix`, or all
  outputs when `delete_all=true`.
- Always list and confirm before deletion.

Default operation outputs live in timestamped directories below `~/.olive-mcp/outputs/`.

## Result fields

Successful operation results may include:

- `device` and `execution_provider`
- `output_models` with model path, ID, type, metrics, inference config, and file size
- `best_model` with the same model details
- `pass_summary` and `total_duration_seconds`
- `input_model_metrics` and `input_model_size_mb`

Failed results include an `error` and may include tailored `suggestions`. Report only fields actually
returned.
