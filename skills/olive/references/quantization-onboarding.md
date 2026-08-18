# Olive Quantization Onboarding

An orientation to Olive's weight-quantization passes for PyTorch/Hugging Face models: what each
pass does, when to reach for it, the config knobs they share, and where to look in the codebase
before making changes. For MoE-specific GPTQ details (per-expert Hessians, fallback thresholds),
see [`moe-gptq.md`](moe-gptq.md). For a worked benchmark example comparing passes end-to-end, see
[`profiling-benchmark-example.md`](profiling-benchmark-example.md).

## Where quantization passes live

PyTorch/Hugging Face weight-quantization passes live in `olive/passes/pytorch/`:

| Pass | Real class / registry name | Module | Calibration data? | Notes |
| --- | --- | --- | --- | --- |
| `Rtn` | `Rtn` | `rtn.py` | No | Round-to-nearest; fastest, data-free, weakest accuracy recovery. |
| `Gptq` | `Gptq` | `gptq.py` | Yes | Layerwise, Hessian-based weight correction using calibration data. |
| AutoGPTQ wrapper | `GptqQuantizer` | `autogptq.py` | Yes | Thin wrapper delegating to the third-party `auto-gptq` library. |
| GPTQModel wrapper | `GptqModel` | `gptqmodel.py` | Yes | Thin wrapper delegating to the third-party `gptqmodel` library. |
| AutoAWQ wrapper | `AutoAWQQuantizer` | `autoawq.py` | Yes | Wraps the third-party `autoawq` library (activation-aware weight quantization). |
| K-quant | `KQuant` | `kquant.py` | Varies | K-quant style block quantization. |

The registry/class name (not a guessed lowercase-of-class-name module path) is what you must pass
to `--pass_name` for `scripts/quantize_and_compare_perplexity.py`, to `olive_config.json`'s
`"passes"` map, and to workflow configs — check `olive/olive_config.json` if you're ever unsure of
the exact registered name for a pass.

ONNX-side quantization passes (post-export, operate on ONNX graphs rather than PyTorch modules)
live separately in `olive/passes/onnx/` (e.g. `rtn_quantization.py`, `hqq_quantization.py`,
`nvmo_quantization.py`, `inc_quantization.py`) — those are a different code path and are out of
scope for this doc, which focuses on the PyTorch-side `Rtn`/`Gptq` family.

Shared logic (quantizer construction, model wrapping/unwrapping, layerwise iteration, save/load)
lives in `olive/passes/pytorch/quant_utils.py`. Read `get_quantizer_config()`,
`prepare_model()`, `run_layerwise_quantization()`, and `finalize()` there before modifying any
pass — almost every pass calls into these four functions and duplicating their logic in a new
pass is very rarely the right move.

## Shared config surface

Every weight-quantization pass accepts a common set of parameters from
`get_quantizer_config()` in `quant_utils.py`:

| Param | Meaning |
| --- | --- |
| `bits` | Quantization bit-width (`PrecisionBits.BITS2/4/8`). |
| `group_size` | Block size for per-group scale/zero-point (`-1` means per-channel/whole-row). |
| `sym` | Symmetric (zero-point fixed at the bit-width's midpoint) vs. asymmetric quantization. |
| `lm_head` | Whether to also quantize the language-model head. |
| `embeds` | Whether to also quantize input embeddings (`Rtn` and `KQuant` only). |
| `overrides` | Per-module overrides for any of the above, keyed by module name pattern. |

`Gptq` additionally exposes `damp_percent` (Hessian damping factor), `desc_act` (activation-order
column permutation, only valid for `group_size=-1`), and `data_config` (calibration dataset — see
below). When `moe=True` it also exposes `moe_fallback_threshold` and
`moe_fallback_min_k_multiple` — see `moe-gptq.md`.

## RTN vs. GPTQ: when to use which

- **RTN** is data-free and much faster than GPTQ (single-digit seconds for a ~1.3B active-param
  model, tens of seconds to ~1-2 minutes for larger multi-billion-parameter MoE models in
  practice — see `profiling-benchmark-example.md` for exact figures), but has the largest
  accuracy regression of the two. Use it as a fast baseline, for environments without a
  calibration dataset, or when the accuracy loss is acceptable for the target use case.
- **GPTQ** uses a calibration dataset to compute per-layer Hessians (`H = sum(x xT)` over
  observed activations) and applies second-order error correction while quantizing each column,
  substantially reducing the accuracy regression relative to RTN at the cost of a much longer
  quantization pass (calibration forward passes + per-layer Cholesky/blockwise-quantize solves).

Empirically (see `profiling-benchmark-example.md` for the full three-model MoE comparison), GPTQ
consistently produced a smaller perplexity regression than RTN on every model tested — but at
roughly 30-80x the quantization wall-time of RTN on the models measured there (single-run,
single-machine timings; treat the exact ratio as illustrative, not a guaranteed multiplier for
every model/hardware combination). Choose RTN when turnaround time matters more than the last bit
of accuracy; choose GPTQ when accuracy matters more and you can afford a longer one-time
quantization pass.

## Calibration data (`data_config`)

`Gptq` (and the other calibration-based passes) accept a `data_config` pointing at a Hugging
Face dataset config, or fall back to Olive's default WikiText-2 calibration set for
`HfModelHandler` inputs if none is given. See `olive/data/config.py` and
`configure-workflows/how-to-configure-data` in the Sphinx docs for how to point at a custom
dataset.

Two split-hygiene facts worth internalizing:

- WikiText-2's `train`/`validation`/`test` are official, pre-defined, non-overlapping Hugging
  Face dataset splits (split by source Wikipedia article) — Olive does not construct or dedupe
  them itself. A handful of near-identical rows across splits (e.g. repeated section-header
  strings like `" = = Career = = "`) are not real leakage; they are trivial boilerplate that
  recurs across unrelated articles.
- Use the `train` split (or a data-appropriate calibration set) for GPTQ calibration and a
  disjoint split (e.g. `test`) for perplexity evaluation. Do not calibrate and evaluate on the
  same rows.

## Before modifying a quantization pass

1. Read `quant_utils.py` fully — most behavior you might think belongs in a specific pass file is
   actually implemented once, shared across passes.
2. Check `test/passes/pytorch/` for the existing test pattern for the pass you're touching (e.g.
   `test_gptq.py`, `test_rtn.py`) before writing new tests; follow the existing
   parametrization/fixture conventions rather than introducing a new test style.
3. If your change affects calibration or fallback behavior for Mixture-of-Experts models
   specifically, read `moe-gptq.md` first — MoE calibration has its own module
   (`moe_calib.py`) and several subtleties (per-expert Hessians, routing skew vs. statistical
   sufficiency) that are easy to get wrong by analogy with the dense-model code path.

If you're adding a brand-new pass rather than modifying an existing one, see the Sphinx guide
`docs/source/how-to/extending/how-to-add-optimization-pass.md` for how to register it in
`olive/olive_config.json` and wire it into the pass-discovery system — that step is not covered
here.
