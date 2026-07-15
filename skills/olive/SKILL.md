---
name: olive
description: Optimize, quantize, fine-tune, convert, and benchmark AI models with Microsoft Olive through the Olive MCP server. Use when a user mentions Olive, ONNX Runtime model optimization, int4 or int8 quantization, RTN, GPTQ, AWQ, HQQ, LoRA or QLoRA training, ONNX graph capture, model benchmarking, or choosing deployment settings for CPU, GPU, or NPU.
license: MIT
compatibility: Requires the Olive MCP server and Python 3.10 or later. Remote Hugging Face models require network access; gated or private models may require a user-provided Hugging Face token.
metadata:
  author: microsoft
  version: "1.0.0"
---

# Olive Model Optimization

Use the tools supplied by the Olive MCP server to perform model work. Tool names may be qualified with an
`olive` namespace; match them by the unqualified names documented below.

If the Olive tools are unavailable, read [the MCP setup guide](references/mcp-setup.md) and help the user
configure the server. Do not pretend an operation ran or substitute an unrelated optimization tool.

## Interaction rules

- Adapt explanations to the user's expertise. Explain precision, quantization, and execution providers in
  plain language unless the user already uses those terms.
- Preserve an explicitly supplied model ID, local path, provider, precision, algorithm, dataset, and output
  path.
- If the request is specific and compatible, run it without asking unnecessary questions.
- If a model is specified but the user only says "optimize," call `detect_hardware`, choose balanced int8
  quantization by default, briefly state why, and run it.
- If the user's goal is genuinely ambiguous, ask one question with complete plain-language plans such as
  smallest model, balanced size and quality, or best GPU quality. Run the selected plan without follow-up
  technical questions.
- If no model is identified, ask for a Hugging Face model ID or local path. When the user wants a
  recommendation instead, ask for the use case and read
  [the model selection guide](references/model-selection.md).

## Choose the operation

Call `detect_hardware` before choosing a provider or precision when the user has not specified the target
hardware. Do not call it for status checks, cancellation, or output management.

| User goal | Tool and starting settings |
| --- | --- |
| Smallest practical CPU model | `quantize` with `precision="int4"` and `algorithm="rtn"` |
| Balanced size and quality | `quantize` with `precision="int8"` |
| Best quality on an NVIDIA GPU | `optimize` with `precision="fp16"` and `provider="CUDAExecutionProvider"` |
| Smallest optimized model on an NVIDIA GPU | `optimize` with `precision="int4"` and `provider="CUDAExecutionProvider"` |
| End-to-end pass scheduling or graph surgery | `optimize` |
| Convert a PyTorch or Hugging Face model to ONNX | `capture_onnx_graph` |
| Fine-tune a text model | `finetune`; prefer `qlora` when memory is constrained, otherwise `lora` |
| Train an SD, SDXL, or Flux LoRA | `finetune` with `data_dir` or `data_name`; diffusion routing is automatic |
| Measure model quality | `benchmark` with tasks relevant to the user's use case |
| Inspect or remove prior results | `manage_outputs` |

Apply these hardware constraints:

- CPU: use `CPUExecutionProvider` and prefer int4 or int8. Never choose fp16 or bf16 for CPU.
- NVIDIA GPU: use `CUDAExecutionProvider`; fp16, int4, and int8 are suitable starting points.
- Windows DirectX 12 GPU: use `DmlExecutionProvider` only when the user requests DirectML or the hardware is
  otherwise known. `detect_hardware` only detects NVIDIA GPUs.
- Qualcomm NPU: use `QNNExecutionProvider` and preserve device-specific settings supplied by the user.

For int4 on CPU, prefer RTN through `quantize`. `optimize` with int4 runs GPTQ calibration and can take more
than 30 minutes on CPU. Use it only when the user explicitly wants GPTQ-quality calibration or has suitable
GPU resources.

Read [the tool reference](references/mcp-tools.md) before supplying non-default parameters.

## Run long operations

`optimize`, `quantize`, `finetune`, `capture_onnx_graph`, and `benchmark` are asynchronous:

1. Call the selected operation and retain its `job_id`.
2. If the call returns an error without a `job_id`, report that error and correct the request instead of
   polling.
3. Call `get_job_status(job_id)` in a loop. Do not add a sleep; the tool long-polls for up to 30 seconds.
4. On every status response, surface `recent_logs` to the user, emphasizing new lines without hiding the real
   Olive output. Summarize the current `phase` in plain language.
5. Continue while the status is `starting`, `setting_up`, or `running`.
6. Stop on `completed`, `error`, `canceled`, or `not_found`, and report the terminal result accurately.

Model download, environment setup, calibration, and training can take 5-30 minutes or longer. If
`new_lines` remains zero and `seconds_since_last_output` keeps increasing across status calls, warn that the
job may be stuck. Never cancel it unless the user asks.

The server allows at most three concurrent jobs. If that limit is reached, wait for an existing job to
finish or ask which job the user wants canceled; do not start repeated retries.

## Handle results

For a successful job, report the fields that are present:

- Best model path and all output model paths
- Input and output file sizes, including the size reduction when both values are available
- Metrics and the evaluation task they belong to
- Device, execution provider, and inference configuration
- Pass summary and total duration

Do not claim that quality or latency improved unless the returned metrics demonstrate it. Run `benchmark`
only when the user requested evaluation or a comparison.

Auto-generated outputs are stored under `~/.olive-mcp/outputs/`. Honor a user-provided `output_path`.

## Handle failures safely

- Show the returned error and actionable `suggestions`; do not produce a success-shaped response.
- For CUDA out-of-memory errors, reduce precision, use RTN int4, reduce batch size, or choose a smaller model
  according to the operation.
- For CPU fp16 or bf16 errors, retry with int4, int8, or fp32.
- For missing dependencies or ONNX Runtime conflicts, explain that the isolated environment may need to be
  recreated. Follow the returned suggestion rather than silently deleting caches.
- For network errors, preserve the request and explain that remote model access is required.
- On a 401, 403, gated-model, or access-denied error, ask for a Hugging Face token and retry with `hf_token`.
  Never echo, log, save, or place the token in any file or command line.

Before any `manage_outputs(action="delete")` call, list the matching outputs, show exactly what would be
deleted, and obtain explicit user confirmation. Treat `prefix` and `delete_all=true` as broad destructive
operations.
