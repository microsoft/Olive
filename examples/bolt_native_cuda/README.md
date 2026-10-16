# Native Bolt CUDA package preservation (no optimization)

This example consumes an **existing local export**, using Olive's supported
`CompositeModel` and `ONNXModel` handlers. It does not download a checkpoint,
export, quantize, rewrite, copy, or execute inference. The output is a reference
`model_config.json` plus a SHA-256 receipt, **not a self-contained deployment
package**. The original package must remain at its original path.

The input is pinned to
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` revision
`c6cc2e8ffa353403213d6b9b18aaf8eb684d705a`, with these components:

```text
package/
  decoder/model.onnx          # native IQ2_XXS, Q8_0 and A16 NVFP4 operators
  decoder/model.onnx.data
  vision_encoder/model.onnx
  vision_encoder/model.onnx.data
  embedding/model.onnx
  embedding/model.onnx.data
  genai_config.json
  tokenizer.json             # and tokenizer/config/chat-template assets
```

## Run

Install Olive with its normal dependencies, plus an ONNX Runtime build containing
CUDA implementations of `com.microsoft::MatMulGGMLIQ2XXS`, `MatMulGGMLQ8_0`,
`MatMulNVFP4A16`, `LinearAttention`, and the required BF16 support. A stock ORT
wheel or merely listing `CUDAExecutionProvider` does **not** establish compatibility.
Configure that build's Python import and shared-library paths before running.
No runtime library is installed or replaced by the example.

```bash
CUDA_VISIBLE_DEVICES=3 python examples/bolt_native_cuda/preserve.py \
  --package-dir /path/to/existing/package \
  --output-dir ./bolt-native-reference
```

The default first creates and releases a CUDA session for each component.
This reads the original external data and may require substantial host/GPU memory.
Missing kernels, unavailable CUDA, or provider initialization fallback cause an
explicit exception. Session creation does **not** check per-node GPU placement,
run inference, prove accuracy, or establish readiness.

For a bounded, runtime-independent inspection that does not materialize external
tensors, explicitly opt out:

```bash
python examples/bolt_native_cuda/preserve.py \
  --package-dir /path/to/existing/package \
  --output-dir ./bolt-native-inspection --inspect-only
```

This records `runtime.status=NOT_CHECKED`. Both modes hash original files by
streaming reads; neither writes inside the source package. Use a new output
directory per run. References are absolute and not relocatable.

`olive_config.json` uses the standard `RunConfig` schema, with package-relative
paths resolved by `preserve.py`. The script validates that configuration and
serializes/reloads the resulting existing handlers. It deliberately does **not**
call `olive run`: this is a model-handler consumption/reference recipe with zero
passes, not an engine optimization workflow or an accuracy evaluator. Additional
root assets, original GenAI config, source pin and graph weight-loading metadata
are propagated through `model_attributes`. ONNX graphs and payloads remain in place
and byte-identical; no metadata is injected into them.

## Supported route and limits

Only the **prepared vision feature** route with **source-bound host prompt
compaction** is described. The caller must supply that host-side implementation;
this example does not implement image preprocessing, feature generation, prompt
compaction, or token generation. `--route ort-genai`, `ort-extensions`, and `cpu`
are explicitly rejected. The graph-faithful `genai_config.json` is retained as
metadata, not converted into a claim that stock ORT GenAI supports this hybrid
decoder or vision retention contract. No fake `image_processor.json` is generated.

Native IQ2_XXS, raw Q8_0 and A16 NVFP4 payloads are already quantized.
Weight-only quantization passes must not be blindly applied to these packed
operators. This recipe rejects **all passes/builds**, including optional
post-export requantization. Any future optimization route requires its own
supported operator scope and numerical qualification; affine INT2 substitution
or full-checkpoint BF16 expansion is not preservation.

Model readiness remains **NOT_ESTABLISHED**, regardless of receipt/session
success. No improvement in size, speed, accuracy, or readiness is claimed.
Numerical acceptance must separately enforce finite results,
NRMSE ≤ 0.03, cosine ≥ 0.999, and exact required tokens, including hidden/cache
gates and multimodal/generation cases. The previous v7 hidden/cache failures
cannot be waived by packaging success; v8 numerical qualification is separate.

## Network-free fixture tests

From the Olive repository root, with pytest installed:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest examples/bolt_native_cuda/test_preserve.py \
  --basetemp=./bolt-recipe-test-artifacts -q
```

The tiny fixtures are constructed using `onnx_ir`, contain opaque packed nodes
and external byte payloads, and are **not executable accuracy fixtures**.
Tests cover schema validation, all component/external-data references and bounds,
bit/graph preservation, metadata round-tripping, explicit unsupported routes,
and missing CUDA/custom-op/provider-fallback errors. Runtime success cases use
mocks and establish only control-flow coverage.
