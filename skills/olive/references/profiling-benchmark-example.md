# Profiling / Benchmark Example: `quantize_and_compare_perplexity.py`

A worked example for `scripts/quantize_and_compare_perplexity.py`, a local validation script that
quantizes a real Hugging Face model with a given Olive pass and reports the perplexity
regression, quantization wall-time, model size, and (for MoE calibration) per-expert fallback
coverage. Read [`quantization-onboarding.md`](quantization-onboarding.md) for background on the
passes being compared and [`moe-gptq.md`](moe-gptq.md) for the MoE-specific fallback mechanism
whose coverage output this script surfaces.

## What the script is (and isn't)

This is a **local, one-off validation tool** — it is not wired into any Olive workflow config or
CI. It exists to answer "does quantizing this real (downloaded) checkpoint produce a sane quality
regression, or does it blow up / silently corrupt the model?", the same class of question that
has previously surfaced real bugs after synthetic-model unit tests had already passed. It is not
a substitute for unit tests, and a single run is a smoke test, not a statistically rigorous
benchmark — see the "Fairness notes" in the script's own module docstring before trusting a
delta from it.

The script is generic: `--pass_name` can be any Olive PyTorch quantization pass (`Gptq`, `Rtn`,
`KQuant`, ...) with any `--pass_config`, and `--model_id` can be any HF model id or local path —
nothing is hardcoded to GPTQ or to MoE.

## Basic usage

```shell
python scripts/quantize_and_compare_perplexity.py \
    --model_id ibm-granite/granite-3.0-1b-a400m-base \
    --pass_name Gptq \
    --pass_config '{"bits": 4, "group_size": 128, "sym": true, "moe": true}' \
    --device cuda:0
```

By default this evaluates perplexity over the **entire** WikiText-2 `test` split and calibrates
GPTQ on Olive's default WikiText-2 `train` split (use `--num_samples` to restrict the eval split
to a prefix for a quicker/dirtier check; see `--help` for calibration-side overrides such as
`--calib_dataset`).

Key flags:

| Flag | Purpose |
| --- | --- |
| `--model_id` | HF model id or local path (required). |
| `--pass_name` | Olive pass class name, e.g. `Gptq`, `Rtn` (default `Gptq`). |
| `--pass_config` | JSON dict of pass config overrides. |
| `--dataset` / `--dataset_config` / `--split` | Eval dataset (default WikiText-2 `test`). |
| `--calib_dataset` / `--calib_dataset_config` | Calibration dataset override for calibrated passes. |
| `--dtype` | Load dtype for both baseline and quantized models (default `auto`, i.e. each checkpoint's native dtype). |
| `--experts_implementation` | Forces a specific MoE experts implementation on both loads for a fair comparison. |
| `--device` | Device for both quantization and eval. |

## Fairness guarantees baked into the script (read before trusting a delta)

- Baseline and quantized models are loaded with the *same* `--dtype` and
  `--experts_implementation`, so a measured perplexity delta is attributable to quantization, not
  to an incidental dtype/backend mismatch between the two loads.
- Reported "weights size" includes **both** an in-memory (parameter-count x dtype-size) figure —
  computed identically for baseline and quantized so the two are apples-to-apples — and a
  separately labeled on-disk size (actual saved weight-file bytes) for the quantized model. Do
  not compare the in-memory baseline figure against the on-disk quantized figure; they are
  different metrics reported side-by-side, not the same metric.
- Calibration sample/token counts are captured by intercepting the actual dataset the pass builds
  internally, not recomputed separately by the script, so they are guaranteed to match what was
  really used to calibrate — and are reported as `n/a` for data-free passes (e.g. RTN).
- "Quantization time" is wall-clock for the entire pass run (load + calibrate + quantize + save),
  not a pure inference benchmark — RTN being much faster than GPTQ is an expected algorithmic
  trade-off, not a regression.

## Worked example: three-way MoE model comparison

The following table was produced by running the script once per (model, pass) pair — six runs
total — with `bits=4, group_size=128, sym=true, moe=true`, full WikiText-2 `train` calibration
(Olive's default post-#2609), and the full WikiText-2 `test` split for eval:

```shell
# Baseline + RTN
python scripts/quantize_and_compare_perplexity.py \
    --model_id <model_id> --pass_name Rtn \
    --pass_config '{"bits": 4, "group_size": 128, "sym": true, "moe": true}' \
    --device cuda:0

# Baseline + GPTQ
python scripts/quantize_and_compare_perplexity.py \
    --model_id <model_id> --pass_name Gptq \
    --pass_config '{"bits": 4, "group_size": 128, "sym": true, "moe": true}' \
    --device cuda:0
```

| Model | Baseline PPL | RTN PPL (Δ) | GPTQ PPL (Δ) | Quant time RTN / GPTQ | Fallback experts |
| --- | --- | --- | --- | --- | --- |
| ibm-granite/granite-3.0-1b-a400m-base | 6.2877 (354,564 tok) | 7.5861 (+1.2984) | 6.9492 (+0.6615) | 8.0s / 653.4s | 3/768 (0.4%) |
| allenai/OLMoE-1B-7B-0924 | 6.6182 (288,720 tok) | 7.1091 (+0.4909) | 6.8937 (+0.2755) | 52.6s / 1484.9s | 21/1024 (2.1%) |
| Qwen/Qwen1.5-MoE-A2.7B | 6.4246 (298,937 tok) | 6.9251 (+0.5005) | 6.6117 (+0.1872) | 100.0s / 2588.9s | 0/1440 (0.0%) |

Calibration set for all GPTQ runs: 128 samples / 262,144 tokens (full WikiText-2 `train` split).
`moe_fallback_min_k_multiple=2.0` (`k=2`, the current default; see `moe-gptq.md`). All
baseline/quantized in-memory weight sizes are equal within each model (fake-quantization
dequantizes back to the original dtype for `transformers` compatibility) — only the *on-disk*
saved size actually shrinks; see the script's own summary output for per-run on-disk figures.

**What this table demonstrates**:

- GPTQ beat RTN on perplexity delta for every model tested (not just on average) — e.g. granite:
  +0.6615 vs. +1.2984; Qwen1.5-MoE: +0.1872 vs. +0.5005 — at the cost of substantially longer
  quantization time (minutes vs. seconds).
- MoE fallback rate is **not** proportional to total expert count — see `moe-gptq.md` for the
  detailed explanation (Qwen1.5-MoE has the most total experts of the three but zero fallbacks;
  the driver is per-model routing skew, not raw expert count).
- Quantization time scales roughly with `num_layers x num_experts` (the per-expert Cholesky solve
  count), not with calibration token count — see `moe-gptq.md` for why a ~4x increase in
  calibration tokens (from a stale `train[:1000]` slice to the full `train` split) only produced
  an ~18% wall-time increase on granite, rather than the naively-expected ~4x.

### `k=1` vs `k=2`: effect of the sufficiency-threshold multiplier

`moe_fallback_min_k_multiple` was raised from `k=1` (the original shipped default, `N=K`) to
`k=2` (`N=2K`, the current default) after a full-team review found the real RTN-vs-GPTQ MSE
crossover under anisotropic activations at 4-bit sits closer to `1.5x-2x K`, not `1x K` (see
`moe-gptq.md`). All three models were rerun with the identical methodology at both settings to
quantify the actual effect:

| Model | GPTQ PPL (Δ) at k=1 | GPTQ PPL (Δ) at k=2 | Fallback experts k=1 | Fallback experts k=2 | GPTQ time k=1 | GPTQ time k=2 |
| --- | --- | --- | --- | --- | --- | --- |
| granite-3.0-1b-a400m-base | 6.9560 (+0.6683) | 6.9492 (+0.6615) | 2/768 (0.3%) | 3/768 (0.4%) | 658.7s | 653.4s |
| OLMoE-1B-7B-0924 | 6.8966 (+0.2784) | 6.8937 (+0.2755) | 10/1024 (1.0%) | 21/1024 (2.1%) | 1499.3s | 1484.9s |
| Qwen1.5-MoE-A2.7B | 6.6117 (+0.1872) | 6.6117 (+0.1872) | 0/1440 (0.0%) | 0/1440 (0.0%) | 2475.6s | 2588.9s |

Baseline and RTN-only numbers are identical between the two runs (RTN never reads
`moe_fallback_min_k_multiple`), confirming no other environment drift between the two benchmark
sessions. Despite roughly doubling the fallback count for granite and OLMoE, **perplexity did not
get measurably worse at `k=2` — it stayed flat or improved slightly**, and quantization time did
not increase (granite/OLMoE were slightly faster; Qwen1.5-MoE's small increase is within normal
run-to-run variance for a ~2,500s job, and it has 0 fallback at both settings so there is no
solve-count difference to explain it). The likely explanation: experts in the `1x-2x K` band have
technically-full-rank but severely ill-conditioned Hessians, so GPTQ's correction there was
already dominated by the damping prior rather than real signal — routing those borderline experts
to RTN at `k=2` removes the risk of an unlucky bad correction without giving up much upside, so
raising the threshold is close to a free win on the models tested here.


## Interpreting a run's console output

Each run prints a `=== SUMMARY ===` block. For a MoE calibration run, also read the per-layer
`MoE coverage [...]` log lines emitted during quantization — each reports the effective fallback
threshold (`max(skew, sufficiency)`), how many experts were starved/unseen, and the observed
min/median/max token counts per expert for that layer. This is the fastest way to tell whether an
unexpectedly high fallback count is caused by routing imbalance in a specific layer or by an
under-sized calibration set overall.

## Tips for running your own comparison

- Start with a small/tiny model (e.g. a `*-tiny-random` HF checkpoint) to smoke-test your
  `--pass_config` before committing GPU time to a multi-billion-parameter model — GPTQ
  quantization time for a large MoE model can run into the tens of minutes.
- Pin `--dtype` explicitly if you need bit-for-bit reproducible baseline numbers across machines;
  `auto` (the default) picks each checkpoint's native dtype, which is normally what you want for
  a realistic baseline but can differ between environments with different default dtype handling.
- If you see a `transformers` warning about token sequence length exceeding
  `max_position_embeddings` during eval-text tokenization, this is expected and harmless — the
  perplexity computation uses a sliding window (`stride`/`max_len`), not a single full-sequence
  forward pass; it does not indicate truncation or a scoring bug.
