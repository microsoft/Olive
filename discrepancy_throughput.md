# ONNX Health Dashboard Throughput Metric Design

**Status:** Draft  
**Owners:** Olive, `onnx-model-discrepancy`, and `test-results` teams  
**Target metric:** Decode throughput in generated tokens per second

## Summary

Add generated-token throughput to the ONNX Health pipeline for the three inference
backends already measured by the discrepancy workflow:

- PyTorch/Transformers
- ONNX Runtime GenAI
- llama.cpp, when llama.cpp measurement is enabled for the run

Olive is the source of truth for metric calculation. The
`onnx-model-discrepancy` repository carries the values into the Cosmos report and
explicitly publishes them through the ONNX Health API. The `test-results`
dashboard adds a Throughput screen under both the PyTorch and Llama.cpp tabs,
including current values and historical trends.

The initial implementation reuses the existing time-to-first-token (TTFT),
time-to-first-N-tokens (TTFN), and `first_n_tokens_timed` measurements. It does
not add another model-generation pass.

## Goals

1. Report comparable decode throughput for PyTorch, ONNX Runtime GenAI, and
   llama.cpp in generated tokens per second.
2. Calculate the metric once, close to measurement, rather than independently in
   the backend and browser.
3. Preserve historical report compatibility. Older reports without throughput
   remain valid and render the metric as unavailable.
4. Make missing or invalid measurements explicit. Do not substitute zero or
   derive a value from unrelated latency fields.
5. Add focused PyTorch and Llama.cpp dashboard views for current values and
   trends.
6. Avoid increasing pipeline inference time in the first release.

## Non-goals

- Request, batch, or concurrent-user throughput.
- Prompt/prefill tokens per second.
- End-to-end output throughput that includes TTFT.
- A throughput pass/fail threshold in the first release.
- Recomputing or backfilling historical Cosmos documents.
- Replacing the existing TTFT, TTFN, total-duration, or speedup metrics.

## Current Data Flow

```text
Olive OnnxDiscrepancyCheck
  -> discrepancy_check_results.json
  -> onnx-model-discrepancy normalization
  -> one Cosmos report document per pipeline run
  -> ONNX Health Function App public-field projection
  -> test-results/docs/onnx-health.html
```

The current Olive pass already records:

- `first_n_tokens_timed`
- `transformers_ttft_s` and `transformers_ttfn_s`
- `genai_ttft_s` and `genai_ttfn_s`
- `llama_cpp_ttft_s` and `llama_cpp_ttfn_s`, when enabled

The backend converts these to camel-case report fields, and the dashboard uses
them for TTFN and speedup views. Throughput should follow the same ownership and
transport path.

## Metric Definition

### Primary metric

The proposed metric is **steady-state decode throughput**:

```text
decode_tokens_per_second = (N - 1) / (TTFN - TTFT)
```

Where:

- `N` is `first_n_tokens_timed`.
- `TTFT` is elapsed time from generation start until the first generated token.
- `TTFN` is elapsed time from generation start until the Nth generated token.
- `N - 1` is used because the interval from the first completed token to the Nth
  completed token contains N - 1 decode intervals.

This definition intentionally excludes initial prompt processing and first-token
latency. TTFT remains the separate measure of prefill and first-token
responsiveness.

### Validity rules

Olive emits throughput only when all of the following are true:

1. `N` is an integer greater than or equal to 2.
2. TTFT and TTFN are finite, non-negative numeric values.
3. `TTFN > TTFT`.
4. The backend generated at least N tokens, so the recorded TTFN represents the
   requested window.

If any condition is false, the throughput field is omitted or `null`. Olive
should log a fixed diagnostic that identifies the backend and failed condition
without including prompts, generated text, paths, or other sensitive values.

Olive must not emit infinity, NaN, a negative value, or zero as a fallback.

### Why not `N / TTFN`

`N / TTFN` is end-to-end output throughput and includes prompt processing and
the first token. That makes it strongly dependent on prompt length and overlaps
with TTFT. It can be added later as a separately named metric if the product
requires it, but it must not be labeled as decode throughput.

### Measurement window

The first release uses the existing `first_n_tokens_timed` value, currently five
tokens in the discrepancy workflow. Every report carries the effective N so the
dashboard can label the window and avoid comparing values calculated with
different N without disclosure.

Five tokens is a short and potentially noisy window. After collecting baseline
data, the teams should evaluate increasing N, running repeated samples, or
reporting percentiles. Such a change must preserve the effective token count in
the report and should be treated as a measurement-method version change.

## Proposed Report Contract

Add the following optional fields to each `models[]` entry:

```json
{
  "throughputTokensTimed": 5,
  "transformersDecodeTokensPerSecond": 14.27,
  "genaiDecodeTokensPerSecond": 41.83,
  "llamaCppDecodeTokensPerSecond": 32.56
}
```

Field rules:

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `throughputTokensTimed` | integer >= 2 | No | Effective N used for all throughput values in the model entry. |
| `transformersDecodeTokensPerSecond` | finite number > 0 | No | PyTorch/Transformers decode throughput. |
| `genaiDecodeTokensPerSecond` | finite number > 0 | No | ONNX Runtime GenAI decode throughput. |
| `llamaCppDecodeTokensPerSecond` | finite number > 0 | No | llama.cpp decode throughput; absent when llama.cpp was not measured. |

`throughputTokensTimed` should be present when any throughput field is present.
It may initially equal `firstNTokensTimed`; the separate name makes the metric
self-describing and permits future measurement changes without overloading TTFN
semantics.

The fields are additive. No top-level report schema version change is required
because model metrics are optional and historical documents are already
supported. If a formal model-metric schema version is introduced later, the
throughput definition should be recorded as version 1 of that schema.

## Repository Changes

### 1. Olive

Primary implementation area:
`olive/passes/onnx/discrepancy_check.py`.

Required changes:

1. Add a small helper such as
   `_decode_tokens_per_second(first_n, ttft_s, ttfn_s)` that applies all validity
   rules and returns `None` when throughput cannot be calculated.
2. In `_compute_final_metrics`, calculate:
   - `transformers_decode_tokens_per_second`
   - `genai_decode_tokens_per_second`
   - `llama_cpp_decode_tokens_per_second`
3. Set `throughput_tokens_timed` when at least one calculation succeeds.
4. Preserve the existing TTFT, TTFN, speedup, token-match, and total-time
   results.
5. Ensure the text, vision-language, speech, and llama.cpp generation paths
   report whether at least N tokens were actually produced. If the current TTFN
   presence already guarantees this for a path, cover that invariant with a
   test rather than inferring it in downstream repositories.
6. Do not add another generation run or new CLI metric selector in phase 1.
   Existing generation metrics already activate the required measurement.

Suggested raw result:

```json
{
  "throughput_tokens_timed": 5,
  "transformers_decode_tokens_per_second": 14.27,
  "genai_decode_tokens_per_second": 41.83,
  "llama_cpp_decode_tokens_per_second": 32.56
}
```

Tests:

- Unit-test the helper with valid values.
- Test `N < 2`, missing values, NaN/infinity, negative values, equal timestamps,
  and `TTFN < TTFT`.
- Test partial availability: for example, PyTorch succeeds while GenAI fails.
- Test generation ending before N leaves throughput unavailable.
- Test text, speech, and llama.cpp result assembly.
- Assert that existing speedup fields are unchanged.

### 2. `onnx-model-discrepancy`

Primary implementation areas:

- `scripts/run_discrepancy_checks.py`
- `scripts/convert_to_dashboard_json.py`
- `azure-function/onnx-health-reports/public-report.js`
- Producer and Function App tests
- `docs/ONNX_HEALTH_REPORT_FORMAT.md`

Required changes:

1. Keep the Olive command unchanged in phase 1. It already requests generation
   measurement through `first_token_20`.
2. Preserve the new snake-case fields when normalizing each
   `discrepancy_check_results.json`.
3. Map them into the proposed camel-case model fields in
   `convert_to_dashboard_json.py`.
4. Validate before upload:
   - token count is an integer >= 2;
   - throughput values are finite numbers > 0;
   - a throughput value cannot be published without
     `throughputTokensTimed`.
5. Add all four fields to the Function App's explicit `MODEL_FIELDS` allowlist.
   The API must not derive throughput from latency values or pass through
   unknown fields.
6. Continue returning reports that do not have throughput.
7. Update the report-format documentation and sample document.

Backend tests:

- Converter maps all fields without changing numeric precision.
- Mixed current and historical result files aggregate successfully.
- Malformed, infinite, negative, and unaccompanied throughput values are
  rejected or omitted according to the repository's report-validation pattern.
- The public report projection includes approved throughput fields.
- Unknown neighboring fields remain private.
- Paginated and non-paginated API responses expose the same metric contract.

### 3. `test-results` frontend

Primary implementation areas:

- `docs/onnx-health.html`
- `docs/ONNX_HEALTH_REPORT_FORMAT.md`
- ONNX dashboard JavaScript contract tests under `tests/`

#### Data ingestion

Extend `flattenReport` with:

- `throughputTokensTimed`
- `transformersDecodeTokensPerSecond`
- `genaiDecodeTokensPerSecond`
- `llamaCppDecodeTokensPerSecond`

Accept only finite positive numbers and an integer token count of at least two.
Invalid data renders as unavailable; it must not be converted to zero. Preserve
the existing API/report sanitization boundary.

#### PyTorch Throughput screen

Add `Throughput` to the PyTorch Focus selector. The screen compares PyTorch with
ONNX Runtime GenAI.

Summary cards:

1. Configurations measured.
2. Median PyTorch decode throughput.
3. Median ONNX decode throughput.
4. ONNX faster than PyTorch, shown as a count and percentage.

Table columns:

| Model | Precision | PyTorch tokens/s | ONNX tokens/s | ONNX/PyTorch ratio | Window | Regression State | Pipeline Error | Last Tested |
|---|---|---:|---:|---:|---:|---|---|---|

The ratio is a presentation-only comparison:

```text
genaiDecodeTokensPerSecond / transformersDecodeTokensPerSecond
```

It should agree with the existing TTFN speedup when both values use the same
window, subject to floating-point rounding. The frontend must not publish this
derived ratio back into the report.

The visualizer adds a `Decode Throughput` metric with PyTorch and ONNX series,
unit `tokens/s`, and the existing model, precision, and date-range controls.

#### Llama.cpp Throughput screen

Add `Throughput` to the Llama.cpp Focus selector. The screen compares llama.cpp
with ONNX Runtime GenAI and appears even when only one backend has a valid value;
missing series render as unavailable rather than suppressing the whole model.

Summary cards:

1. Configurations measured.
2. Median llama.cpp decode throughput.
3. Median ONNX decode throughput.
4. ONNX faster than llama.cpp, shown as a count and percentage.

Table columns:

| Model | Precision | Llama.cpp tokens/s | ONNX tokens/s | ONNX/Llama.cpp ratio | Window | Regression State | Pipeline Error | Last Tested |
|---|---|---:|---:|---:|---:|---|---|---|

The visualizer adds a `Decode Throughput` metric with llama.cpp and ONNX series.

#### Presentation rules

- Label the unit as `tokens/s`; do not use ambiguous `TPS` without a tooltip.
- Display two decimal places in tables and tooltips.
- Use the effective token window in the subtitle, for example
  `Decode throughput over tokens 2-5`.
- If loaded reports contain multiple window sizes, label each row and do not
  aggregate their medians into one summary. Prefer the newest report's window
  and show a mixed-window warning, or group the summary by N.
- Sort throughput columns numerically with missing values last.
- Use `-` or the dashboard's established em dash for missing values.
- Do not color a value as a regression until a throughput regression policy is
  approved.

Frontend tests:

- Current reports render both Throughput focus screens.
- Historical reports without the fields continue to render all existing views.
- Partial backend data renders the available value and an unavailable peer.
- Invalid numeric values do not render or affect summaries.
- Numeric sorting handles decimals and missing values.
- PyTorch and Llama.cpp visualizers select and chart throughput.
- Mixed N values are labeled and never silently aggregated.

## Regression Policy

Throughput is informational during the initial rollout. Existing regression
state remains driven by the current backend policy; the frontend must not infer
a new regression from a throughput delta.

After enough stable runs are collected, add a separate policy proposal that
defines:

- minimum baseline run count;
- grouping key, including model, precision, provider/device, architecture, and
  throughput window N;
- acceptable relative degradation;
- noise handling, such as median-of-runs or EWMA;
- hardware and software version boundaries that invalidate a baseline.

A likely rule is a relative decrease from a historical baseline, but the
threshold must be selected from observed variance rather than chosen in this
implementation.

## Rollout Plan

Deployment order is important because the API uses an explicit field allowlist:

1. **Olive:** merge metric calculation and unit tests.
2. **`onnx-model-discrepancy` producer:** update normalization, validation,
   report documentation, and tests.
3. **`onnx-model-discrepancy` Function App:** deploy the public-field projection
   before expecting the frontend to receive the values.
4. **Pipeline canary:** run a small CPU model set and, where available, a
   llama.cpp-enabled model. Inspect the raw Olive result, normalized result,
   Cosmos document, and unauthenticated API response.
5. **`test-results`:** deploy the frontend screens after the API response is
   verified.
6. **Full pipeline:** enable the values for all existing model and hardware
   stages without a historical backfill.

Each step is backward compatible. Deploying Olive first produces fields that an
older backend ignores. Deploying the backend first leaves the new fields absent.
Deploying the frontend first shows no throughput data but must not break other
screens.

## Verification

For one model configuration, verify the same values at every boundary:

1. Recalculate `(N - 1) / (TTFN - TTFT)` from the raw Olive result.
2. Confirm the calculated raw throughput matches within floating-point
   tolerance.
3. Confirm the normalized model result and Cosmos model entry preserve it.
4. Confirm the Function App returns only the approved camel-case fields.
5. Confirm the PyTorch and Llama.cpp screens display the value, token window,
   sort order, and historical chart correctly.

Suggested floating-point test tolerance:

```text
absolute error <= 1e-9
```

The UI rounds only for presentation and retains the API number for sorting and
charting.

## Observability

The pipeline should summarize, per run:

- model configurations with PyTorch throughput;
- model configurations with ONNX throughput;
- model configurations with llama.cpp throughput;
- requested measurements that were unavailable;
- invalid timing windows rejected by Olive or backend validation.

Do not log prompts, generated tokens, model output text, internal paths, or raw
commands as part of throughput diagnostics.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Five-token windows are noisy. | Ship as informational, retain N in every report, evaluate a longer window after collecting variance data. |
| TTFT and TTFN are measured differently across backend adapters. | Calculate in Olive, test each generation path, and require the same start and completion semantics. |
| A model emits EOS before N. | Do not publish throughput for that backend; expose partial availability. |
| Historical and current reports have different fields. | Keep all new fields optional and test mixed report sets. |
| Future changes to N create misleading trends. | Label N, separate mixed windows, and include N in any future regression baseline key. |
| Browser-derived values drift from producer semantics. | Treat Olive fields as authoritative; derive only display ratios from published throughput. |
| API allowlist silently hides new data. | Deploy and test the Function App projection before the frontend rollout. |

## Open Questions

1. Should phase 2 increase the throughput window from five to 20 or more tokens?
2. Should repeated generation samples report median and percentile throughput?
3. Should throughput be measured for models that stop before N using their
   actual token count, or remain unavailable as proposed?
4. Should the future regression policy compare absolute tokens/s, relative
   backend ratios, or both?
5. Should the dashboard expose end-to-end output thVroughput as a separate metric
   in addition to decode throughput?

## Acceptance Criteria

- Olive emits valid, finite decode-throughput values for each successfully
  measured backend using the documented formula.
- The backend validates, maps, stores, and publicly projects the optional fields.
- Historical reports remain valid without migration.
- The PyTorch tab has a Throughput focus screen comparing PyTorch and ONNX.
- The Llama.cpp tab has a Throughput focus screen comparing llama.cpp and ONNX.
- Both screens include summaries, sortable tables, measurement-window labels,
  and historical charts.
- Missing or invalid data is explicit and never represented as zero.
- No throughput regression status is introduced in the initial release.
- Unit and contract tests cover calculations, transport, compatibility, and UI
  rendering.
