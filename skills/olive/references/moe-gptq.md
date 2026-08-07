# MoE GPTQ Onboarding

How Olive's `Gptq` pass (`olive/passes/pytorch/gptq.py`) quantizes Mixture-of-Experts (MoE)
models, why it needs a dedicated calibration path (`olive/passes/pytorch/moe_calib.py`), the
dual fallback-threshold design, and what the empirical validation from a three-model benchmark
showed. Read [`quantization-onboarding.md`](quantization-onboarding.md) first for the general
RTN/GPTQ context if you haven't already. For how to reproduce the benchmark numbers cited below,
see [`profiling-benchmark-example.md`](profiling-benchmark-example.md).

## Why MoE needs its own calibration path

Standard (dense-model) GPTQ calibration hooks a single `nn.Linear` per layer and accumulates one
Hessian from whatever activations flow through it. Fused MoE architectures don't have a per-expert
`nn.Linear` — routing happens *inside* a single "experts" module's forward call, so a plain
`register_forward_hook` on that module sees one undifferentiated activation batch with no way to
attribute individual rows to the expert that actually processed them.

`moe_calib.py` solves this by hooking into transformers' own experts-implementation registry
(`ALL_EXPERTS_FUNCTIONS` / `@use_experts_implementation`, requires `transformers >= 5.0`):
Olive registers one generic recording implementation and points the model at it for the duration
of calibration. Every decorated experts class dispatches through it uniformly, so this file
contains no per-architecture branching for the recording logic itself — only an allow-list (see
below) of which architectures it has been verified safe for.

## The K-last layout requirement

GPTQ's `(K, K)` Hessian math assumes the weight's contraction dimension (`K`, the input-feature
dimension) is the last dimension of the fused weight tensor: `(num_experts, out_features, K)`.
Olive's MoE support is **allow-listed**, not best-effort, because getting this wrong silently
mis-quantizes the model rather than erroring:

- `SUPPORTED_MOE_EXPERTS_CLASSES` in `moe_calib.py` lists the exact experts class name (not just
  `model_type`) verified to (a) use `(num_experts, out, in)` — K-last — fused weight storage, and
  (b) carry the `@use_experts_implementation` decorator. Currently verified: DeepSeek-V3,
  GraniteMoE, Jamba, Mixtral, OLMoE, Phi-MoE, Qwen2-MoE, Qwen3-MoE (against transformers 5.14.1).
- The class name is checked *in addition to* `model_type` specifically so a spoofed or mismatched
  `config.model_type` string cannot smuggle an unverified experts module past the allow-list.
- **Transposed-layout architectures** (`gpt_oss`, `llama4`, `aria`) store `(num_experts, in, out)`
  — K is not last. These are refused with a clear error rather than silently mis-quantized.
  Supporting them requires a layout-normalization step that is deliberately out of scope for the
  current implementation — if you need this, that normalization (transpose before Hessian
  accumulation, transpose back before save) is the right place to start, not extending the
  allow-list to include them as-is.

If you're adding support for a new MoE architecture: verify its fused weight layout and decorator
support against the currently-installed `transformers` version before adding it to
`SUPPORTED_MOE_EXPERTS_CLASSES` — do not assume a new architecture matches by analogy to an
existing allow-listed one.

## Per-expert fallback: two independent conditions

Not every expert in a calibration run receives enough routed tokens to compute a well-formed
Hessian. When an expert's calibration signal is inadequate, `Gptq` falls back to quantizing that
expert with RTN instead of GPTQ — RTN quantization from a poorly-calibrated Hessian is never
worse than forcing GPTQ through a nearly-empty one, since GPTQ's correction degenerates toward
the damping prior anyway once the Hessian is under-determined.

An expert falls back if it fails **either** of two independent conditions (an OR-gate,
implemented as `LayerCoverage.threshold = max(skew_threshold, k_threshold)` in `moe_calib.py`
and mirrored in `Gptq._process_moe_param` in `gptq.py`):

1. **Routing skew** (`moe_fallback_threshold`, default `0.005` = 0.5%, matching GPTQModel's
   default): is this expert under-served *relative to its peers* in this same calibration run?
   Computed as `fallback_threshold * tokens_seen_by_this_layer`. This condition is
   **scale-invariant** — if you 10x the calibration set, this expert's absolute token count also
   scales ~10x, so the skew ratio is roughly unchanged. It catches routing imbalance but not
   under-calibration of the whole run.
2. **Statistical sufficiency** (`moe_fallback_min_k_multiple`, default `1.0`): does this expert
   have *enough absolute samples* to produce a well-formed `(K, K)` Hessian at all? An expert's
   Hessian `H = sum(x xT)` accumulated from `N` routed tokens satisfies `rank(H) <= N`, so
   `N < K` makes `H` **provably** singular — not just noisy. `N = K` is the information-theoretic
   floor (every direction in the K-dimensional input space has at least one data point), derived
   from first principles rather than an externally-cited tuned hyperparameter (a search of
   available MoE-quantization literature found no numeric "safe multiple of K" recommendation —
   only qualitative guidance). This condition is **absolute**: more calibration data always helps
   it directly, unlike the routing-skew condition.

### Why both conditions, and why this was a design-vs-implementation gap

The original design doc (`gptq_moe_design.md`, section on fallback thresholds) settled on
K-multiple sufficiency as the fallback trigger, treating the routing-skew percentage as
diagnostic/reporting-only. The first shipped implementation, however, only implemented the
skew-percentage condition (`DEFAULT_MOE_FALLBACK_THRESHOLD = 0.005`) from its very first commit —
the design's settled K-multiple decision was never actually implemented. This was not a later
regression; it was a design decision that was reconciled only during later benchmark validation
work, which added `moe_fallback_min_k_multiple` and made both conditions gate the fallback (a
stricter combination than the design doc's own final call, which only used skew as the gate).

**Empirical evidence the two conditions are not redundant** (OLMoE-1B-7B-0924, full WikiText-2
`train` calibration, 262,144 calibration tokens): layer 2, expert 5 received `N=1331` tokens.
- Skew threshold: `0.005 * 262,144 = 1310.7` — `1331 > 1310.7`, so the **skew-only** check would
  **not** flag this expert (it looks "fair" relative to its peers).
- Sufficiency threshold: `1.0 * K = 1.0 * 2048 = 2048.0` (K=2048 for `gate_up_proj`) —
  `1331 < 2048.0`, so the **sufficiency** check **does** flag it: this expert's Hessian is
  provably rank-deficient (`N < K`) regardless of how "fair" its share of tokens looks.

This is direct confirmation that an expert can pass the routing-skew check while still having a
mathematically inadequate Hessian — validating the dual-condition (OR-gate) design over either
condition alone.

## Coverage logging and diagnostics

Every MoE layer's calibration coverage is logged via `LayerCoverage.summary()` (see
`moe_calib.py`), which reports, per layer: number of experts covered/starved/unseen, the combined
effective threshold (`max(skew, sufficiency)`), and the min/median/max observed token count per
expert. Read this log when investigating an unexpectedly high fallback count for a given model —
it will tell you whether the fallback is driven by routing skew (uneven distribution across
experts) or raw insufficiency (the whole calibration set is too small for this model's K).

## What the three-model benchmark showed about fallback rates

Fallback rate does **not** scale with the total number of experts in a model. From a benchmark
across three MoE models (bits=4, group_size=128, sym=true, full WikiText-2 `train` calibration —
see `profiling-benchmark-example.md` for the full table and reproduction steps):

| Model | Experts/layer x layers | Total experts | Fallback experts |
| --- | --- | --- | --- |
| granite-3.0-1b-a400m-base | 32 x 24 | 768 | 2 (0.3%) |
| OLMoE-1B-7B-0924 | 64 x 16 | 1024 | 10 (1.0%) |
| Qwen1.5-MoE-A2.7B | 60 x 24 | 1440 | 0 (0.0%) |

Qwen1.5-MoE has the *most* total experts (1440, driven by having more layers, not more
experts-per-layer than OLMoE) but *zero* fallbacks, while OLMoE has fewer total experts but the
*highest* fallback rate. Since calibration tokens are split **per layer** among that layer's
experts, the relevant comparison is experts-per-layer (~60-64, comparable across both models),
not the total. The actual driver is **per-layer routing skew**: OLMoE's per-layer logs showed
genuine imbalance (e.g. the layer-2-expert-5 case above, min tokens well below median in that
layer), while Qwen1.5-MoE's per-layer logs showed min tokens/expert consistently well above the
sufficiency threshold across all 24 layers (4858-11330), with no layer ever showing a
significantly under-routed expert.

**Takeaway**: read fallback count as a diagnostic of *that specific model's router balance on
that specific calibration domain*, not as a function of how many experts the model has. Two
models with similar experts-per-layer counts can have very different fallback rates purely
because their trained routers distribute tokens differently (this is plausibly related to
architecture differences like shared-expert-plus-top-k gating and the load-balancing auxiliary
loss used at pretraining time, but this benchmark did not directly inspect router logits to
confirm that hypothesis).

## Quantization wall-time: calibration-set size matters less than you'd expect

A common intuition is "bigger calibration set → proportionally longer GPTQ quantization." For
MoE models this is **not** the dominant effect. GPTQ per-layer cost splits into two phases with
very different scaling behavior:

1. **Hessian accumulation** (forward passes over calibration data) — scales ~linearly with the
   number of calibration tokens.
2. **Per-expert GPTQ solve** (Cholesky decomposition + blockwise quantization on each expert's
   `(K, K)` Hessian) — a **fixed cost** independent of how many tokens built that Hessian; it
   depends only on `K` and the number of (expert, parameter) pairs to solve.

On granite-3.0-1b-a400m-base, increasing the calibration set from `train[:1000]` (~61,816 tokens)
to the full `train` split (262,144 tokens, ~4.24x more) only increased quantization wall-time by
~18% (658.7s vs. ~558s), because phase 2 (1536 fixed-cost Cholesky/quantize solves for this
model: 24 layers x 32 experts x 2 params) dominates total time for a model with a modest
active-parameter count and many experts. This ratio is model-specific — for a model with a much
larger active-parameter count (slower forward pass) or fewer experts (solve phase less dominant),
the balance could shift back toward calibration-size-linear scaling — but the two-phase mechanism
itself generalizes. Practically: total quantization time scales roughly with
`num_layers x num_experts` (see the three-model table above: 768/1024/1440 solves roughly
tracking 658.7s/1499.3s/2475.6s), not with calibration token count.

## Before touching MoE GPTQ code

1. Read `moe_calib.py`'s module docstring and `SUPPORTED_MOE_EXPERTS_CLASSES` first — the
   allow-list is a deliberate fail-closed safety mechanism, not an oversight to work around.
2. If you're changing fallback-threshold behavior, keep both conditions in mind — they measure
   different things (relative skew vs. absolute sufficiency) and are not interchangeable.
3. Check `test/passes/pytorch/test_gptq.py` and any MoE-specific test file for the existing test
   conventions (parametrization over architectures, fixture patterns) before adding new tests.
4. Validate any threshold or calibration change against a real model, not just synthetic/tiny
   test fixtures — the dual-condition disagreement case above only showed up on a real model
   (OLMoE) with real routing behavior; tiny random-weight test models are unlikely to reproduce
   realistic routing skew.
