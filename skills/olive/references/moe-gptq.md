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
expert with RTN instead of GPTQ. The design doc's rationale (`gptq_moe_design.md`) is that this
should never be *worse* than plain RTN for that expert, since GPTQ's correction degenerates
toward the damping prior once the Hessian is under-determined — but note this is a working design
policy, not a formally proven guarantee: the design doc itself flags that a damped, rank-deficient
Hessian does not mathematically reduce to exactly RTN (damping reweights the null directions
rather than eliminating their influence, and the layerwise corrections stay coupled across
columns). Treat "never worse than RTN" as the intended, empirically-motivated behavior rather than
a theorem.

An expert falls back if it fails **either** of two independent conditions (an OR-gate,
implemented as `LayerCoverage.threshold = max(skew_threshold, k_threshold)` in `moe_calib.py`
and mirrored as `N < skew_threshold or N < sufficiency_threshold` in `Gptq._process_moe_param` in
`gptq.py`):

1. **Routing skew** (`moe_fallback_threshold`, default `0.005` = 0.5%, matching GPTQModel's
   default): is this expert under-served *relative to its peers* in this same calibration run?
   Computed as `fallback_threshold * tokens_seen_by_this_layer`. This condition is
   **scale-invariant** — if you 10x the calibration set, this expert's absolute token count also
   scales ~10x, so the skew ratio is roughly unchanged. It catches routing imbalance but not
   under-calibration of the whole run. Note this threshold is only ever binding when
   `fallback_threshold * tokens_seen < K` for the parameter being checked — at Olive's default
   262,144-token calibration budget, that means the skew condition can only fire for `K <~ 1311`;
   for larger `K` values it is structurally dominated by the sufficiency condition below (see the
   OLMoE example) and never actually gates anything on its own.
2. **Statistical sufficiency** (`moe_fallback_min_k_multiple`, default `2.0`): does this expert
   have *enough absolute samples* to produce a well-formed `(K, K)` Hessian at all? An expert's
   Hessian `H = sum(x xT)` accumulated from `N` routed tokens satisfies `rank(H) <= N`, so
   `N < K` is *necessary* for `H` to be singular but not by itself sufficient to prove it (`N`
   linearly-dependent samples at `N >= K` can still leave `H` rank-deficient) — practically,
   `N < K` (`k=1`) is only the bare-minimum-rank floor, not the point at which the Hessian is
   actually well-conditioned. A full-team review of this pass measured the real RTN-vs-GPTQ MSE
   crossover under anisotropic activations at 4-bit to sit closer to `N` in the `1.5x-2x K` range:
   at the `k=1` floor, GPTQ was measured ~4-6% *worse* than RTN on average for experts in that
   band. `moe_fallback_min_k_multiple` was therefore raised from the original `k=1` default to
   `k=2` (`N = 2K`) as a conservative, empirically-motivated policy choice past that crossover —
   not a proven sufficient threshold. No external MoE-quantization literature search turned up a
   specific "safe multiple of K" recommendation, only qualitative guidance; `k=2` is the repo's
   own measured choice. This condition is **absolute**: more calibration data always helps it
   directly, unlike the routing-skew condition.

Note also that fallback is decided **per (expert, parameter)**, not per expert as a whole: a fused
MoE layer typically has two quantizable parameters (`gate_up_proj`, `down_proj`), and they can
have different `K` (input-feature dimension), so the *same* expert can pass the sufficiency check
for one parameter and fail it for the other. "N/M fallback experts" in a coverage summary counts
starved-or-unseen occurrences for the parameter(s) actually tracked in the coverage report, not
necessarily every quantizable parameter for that expert — check the per-layer log for the
specific parameter dimension if you need to know exactly which weights fell back.

### Why both conditions, and why this was a design-vs-implementation gap

The original design doc (`gptq_moe_design.md`) settled on K-multiple sufficiency as the *sole*
fallback trigger, treating the routing-skew percentage as diagnostic/reporting-only ("skew =
diagnostic (report); sufficiency = trigger (fallback)"). The first shipped implementation,
however, only implemented the skew-percentage condition (`DEFAULT_MOE_FALLBACK_THRESHOLD =
0.005`) from its very first commit — the design's settled K-multiple decision was never actually
implemented. This was not a later regression; it was a design decision that was reconciled only
during later benchmark validation work, which added `moe_fallback_min_k_multiple` and made
*both* conditions gate the fallback via an OR-gate — a stricter combination than the design doc's
own final call, which gated on sufficiency alone and used skew only for reporting.

**Empirical evidence the two conditions are not redundant** (OLMoE-1B-7B-0924, full WikiText-2
`train` calibration, 262,144 calibration tokens reaching the layer, `k=1` i.e.
`moe_fallback_min_k_multiple=1.0`, the value in effect when this specific case was first found):
layer 2, expert 5 received `N=1331` tokens for its `gate_up_proj` parameter (K=2048).
- Skew threshold: `0.005 * 262,144 = 1310.7` — `1331 > 1310.7`, so the **skew-only** check would
  **not** flag this expert. This is not because the expert is genuinely well-served: OLMoE routes
  top-8-of-64, so a "fair share" of tokens for one expert is `262,144 * 8/64 = 32,768` — the
  1,331 tokens this expert saw is only ~4% of fair share (~24x under-routed). The skew threshold
  simply doesn't bind at this `K`, for the structural reason noted above (0.5% of 262,144 =
  1310.7 < K=2048), not because the expert looks reasonably calibrated.
- Sufficiency threshold at `k=1`: `1.0 * K = 1.0 * 2048 = 2048.0` — `1331 < 2048.0`, so the
  **sufficiency** check **does** flag it: this expert's Hessian is necessarily rank-deficient
  (`N < K`).

This confirms the two conditions are not redundant in general (there exist real experts the
sufficiency check catches that skew alone would not), though this specific benchmark only produced
a case in the "sufficiency fires, skew doesn't" direction — it does not exercise a case where skew
would fire but sufficiency wouldn't, so it validates the OR-gate's usefulness over
skew-alone, but does not by itself demonstrate a case where dropping the sufficiency-alone design
(skew as report-only) would have been wrong in the opposite direction.

With the default later raised to `k=2` (`moe_fallback_min_k_multiple=2.0`, see the note above),
this same OLMoE layer/expert is caught even more clearly: the sufficiency threshold becomes
`2.0 * 2048 = 4096.0`, so `1331 < 4096.0` by a wider margin, and the same rerun at `k=2` also
newly catches several additional experts across other layers whose `N` fell in the `1x-2x K`
"no man's land" (e.g. layer 8 expert 18/48/55 with `N` in the 2445-3772 range against
`K=2048` — these pass at `k=1` but fail at `k=2`). See "What the three-model benchmark showed"
below for the aggregate before/after `k=1` vs `k=2` comparison.

## Coverage logging and diagnostics

Every MoE layer's calibration coverage is logged via `CoverageReport.log_summary()` (built from
per-layer `LayerCoverage.format()` strings; see `moe_calib.py`), which reports, per layer: number
of experts covered/starved/unseen, the combined effective threshold (`max(skew, sufficiency)`),
and the min/median/max observed token count per expert. Read this log when investigating an
unexpectedly high fallback count for a given model — it will tell you whether the fallback is
driven by routing imbalance across experts or raw insufficiency (the whole calibration set too
small for this model's K), and at which specific `K` (i.e. which parameter) the threshold bound.

## What the three-model benchmark showed about fallback rates

Fallback rate does **not** scale with the total number of experts in a model. From a benchmark
across three MoE models (bits=4, group_size=128, sym=true, full WikiText-2 `train` calibration,
`k=2` i.e. `moe_fallback_min_k_multiple=2.0` — see `profiling-benchmark-example.md` for the full
table and reproduction steps):

| Model | Experts/layer x layers | Total experts | Fallback experts (k=2) |
| --- | --- | --- | --- |
| granite-3.0-1b-a400m-base | 32 x 24 | 768 | 3 (0.4%) |
| OLMoE-1B-7B-0924 | 64 x 16 | 1024 | 21 (2.1%) |
| Qwen1.5-MoE-A2.7B | 60 x 24 | 1440 | 0 (0.0%) |

Qwen1.5-MoE has the *most* total experts (1440, driven by having more layers, not more
experts-per-layer than OLMoE) but *zero* fallbacks, while OLMoE has fewer total experts but the
*highest* fallback rate. Since calibration tokens are split **per layer** among that layer's
experts, the relevant comparison is experts-per-layer (~60-64, comparable across both models),
not the total. The proximate cause is genuine routing imbalance for the affected OLMoE experts
(e.g. the layer-2-expert-5 case above, ~24x under fair share), but note the *fallback condition
that actually fires* for these K=2048 cases is sufficiency, not skew (skew is structurally inert
at this K, per the note above) — "routing skew" describes the underlying router behavior, while
"sufficiency" is the specific check that catches it in the current implementation. Qwen1.5-MoE's
per-layer logs showed min tokens/expert consistently well above the `k=2` sufficiency threshold
across all 24 layers, with no layer ever showing a significantly under-routed expert.

Raising the sufficiency multiplier from `k=1` to `k=2` roughly doubled the fallback count for
granite (2->3) and OLMoE (10->21) as expected — more experts in the `1x-2x K` "no man's land" are
now caught — while Qwen1.5-MoE's fallback count stayed at 0 in both runs (its router is well
enough balanced on this calibration set that no expert falls below even the `k=2` threshold).
Despite the higher fallback count, **GPTQ perplexity did not get measurably worse at `k=2` vs
`k=1`** (see `profiling-benchmark-example.md`'s `k=1` vs `k=2` comparison table) — consistent with
the `1x-2x K` band being a region where GPTQ wasn't reliably beating RTN in the first place, so
routing those experts to RTN removes downside risk at effectively no aggregate cost.

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

On granite-3.0-1b-a400m-base, increasing the calibration set from a stale `train[:1000]` slice
(~61,816 tokens, a single unreplicated prior run, not independently re-verified) to the full
`train` split (262,144 tokens, ~4.24x more) only increased quantization wall-time by ~18%
(658.7s at `k=1` vs. ~558s in that single comparison) — consistent with phase 2 (1,536
fixed-cost Cholesky/quantize solves for this model: 24 layers x 32 experts x 2 params) dominating
total time for a model with a modest active-parameter count and many experts. This ratio is
model-specific and based on a single before/after comparison, not repeated runs — for a model
with a much larger active-parameter count (slower forward pass) or fewer experts (solve phase
less dominant), the balance could shift back toward calibration-size-linear scaling. Treat the
two-phase *mechanism* as the durable takeaway; treat the specific "~18% for ~4.24x more data"
ratio as one illustrative data point, not a general formula.

The three-model table's wall-times at `k=1` (658.7s / 1499.3s / 2475.6s for 1,536 / 2,048 / 2,880
total solves) are directionally consistent with solve-count-driven scaling, but with only three
data points where solve count, `K`, and total parameter volume all increase together, this
benchmark cannot cleanly separate "cost driven by number of solves" from "cost driven by total
active parameter volume" as the dominant covariate — both plausibly contribute. Don't treat
`num_layers x num_experts` as a validated predictive formula from this data alone; treat it as the
mechanistically-motivated hypothesis the two-phase model above suggests. Rerunning the same three
models at `k=2` (653.4s / 1484.9s / 2588.9s) showed wall-time essentially unchanged from `k=1`
despite the higher fallback count — consistent with the fixed-cost-per-solve model, since raising
the sufficiency multiplier slightly *reduces* the number of GPTQ solves actually performed (more
experts skip straight to RTN), so total time trends flat-to-down rather than up.

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
