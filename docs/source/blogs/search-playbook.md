# Taming the Search Space: How Olive Finds the Best Quantization Recipe for a PyTorch Model

Quantizing a model is easy. Quantizing it *well* is not. Every quantization algorithm exposes a handful of knobs — bit width, group size, symmetric vs. asymmetric, which layers to protect — and the "best" combination is almost never the default one. It's model-specific, and the only reliable way to find it is to try many combinations and measure.

That's exactly what Olive's **search** capability is for. This post walks through how it works using a realistic example: optimizing a PyTorch/Hugging Face model with [`SelectiveMixedPrecision`](../reference/pass.rst#selectivemixedprecision) followed by [GPTQ](../reference/pass.rst#gptq) quantization. Along the way, we'll see how quickly a handful of options turns into a combinatorial explosion — and how Olive keeps that explosion manageable.

```{note}
GPTQ is just this post's running example. Everything here about search spaces,
joint vs. pass-by-pass execution, samplers, and one-parameter-at-a-time sweeps
applies just as much to Olive's other PyTorch quantization passes — RTN, AWQ,
SpinQuant, QuaRot, and others — since they're all just passes with their own
configuration/search parameters (where declared), chained together
(optionally behind ``SelectiveMixedPrecision``) the same way.
```

---

## Why search, not just configuration?

A typical Olive config lets you pin every pass parameter to a fixed value:

```json
{ "type": "Gptq", "bits": 4, "group_size": 32, "sym": false }
```

That runs GPTQ exactly once, with exactly those values. But how do you know ``group_size: 32`` is better than ``128`` for *this* model? You don't, until you try both — and try them together with every other parameter that might interact with it.

Instead of fixing values, Olive lets you declare a **search parameter**: a discrete set of candidate values for Olive to explore automatically.

```json
{ "type": "Gptq", "bits": "SEARCHABLE_VALUES", "group_size": "SEARCHABLE_VALUES" }
```

In practice you don't even need to write this — most quantization passes already ship with sensible ``search_defaults`` built in. You just tell Olive *how* to search (how many trials, which sampler, joint vs. staged) and it does the rest.

---

## Step 1: `SelectiveMixedPrecision` — pick your protected layers

[`SelectiveMixedPrecision`](../reference/pass.rst#selectivemixedprecision) doesn't quantize anything itself — it annotates the model with a plan for *which layers get extra precision* before a later pass (like GPTQ) quantizes it. This is the mechanism behind mixed-precision recipes: keep a few sensitive layers (e.g. the LM head) at higher precision while everything else drops to low-bit.

Its searchable options are:

| Parameter | Candidate values | What it controls |
|---|---|---|
| `algorithm` | ``high_precision_lm_head``, ``high_precision_mlp_down``, ``high_precision_mlp_down_qkv``, ``snr``, ``snr_relative``, ``iqe``, ``iqe_relative``, ``kld_gradient`` | Which layers get promoted to high precision — a fixed heuristic (by layer position) or a sensitivity-score-based selection (SNR, IQE, or KL-divergence gradient) |
| ``bits`` | `2`, `4`, `8` | Default precision for non-protected layers |
| ``group_size`` | `-1`, `16`, `32`, `64`, `128` | Quantization block size (score-based algorithms only) |
| ``sym`` | `True`, `False` | Symmetric vs. asymmetric quantization (score-based algorithms only) |
| ``high_bits`` | `8`, `16` | Precision used for selected high-precision layers |
| ``high_group_size`` | `None`, `-1`, `16`, `32`, `64`, `128` | Optional block size override for high-precision layers |
| ``high_sym`` | `None`, `True`, `False` | Optional symmetry override for high-precision layers |
| ``ratio`` | `0.5`, `0.8`, `0.9`, `0.95` | Fraction of default-precision parameters used by score-based algorithms; required for score-based modes |

Eight parameters, each with a small number of options — looks harmless. But the **search space is the Cartesian product** of every parameter's candidates when you leave them unset and search is enabled:

$$
8 \text{ (algorithm)} \times 3 \text{ (bits)} \times 5 \text{ (group\_size)} \times 2 \text{ (sym)} \times 2 \text{ (high\_bits)} \times 6 \text{ (high\_group\_size)} \times 3 \text{ (high\_sym)} \times 4 \text{ (ratio)} = 5{,}760 \text{ configurations}
$$

That is the full search space if you leave the high-precision knobs and `ratio` unset. In the examples below, we pin those fields explicitly so the documented search space stays at the simpler 240-point baseline for the default heuristic configuration and to avoid invalid score-based searches that omit a valid `ratio`.

---

## Step 2: Add GPTQ — the matrix doesn't add, it multiplies

Now chain [GPTQ](../reference/pass.rst#gptq) after `SelectiveMixedPrecision` to actually quantize the annotated model. GPTQ has its own set of searchable parameters:

| Parameter | Candidate values | What it controls |
|---|---|---|
| ``bits`` | `2`, `4`, `8` | Quantization bit width |
| ``group_size`` | `-1`, `16`, `32`, `64`, `128` | Quantization block size |
| ``sym`` | `True`, `False` | Symmetric vs. asymmetric quantization |
| ``lm_head`` | `True`, `False` | Whether to quantize the LM head |
| ``damp_percent`` | `0.001`, `0.01`, `0.1` | Hessian damping factor used during weight correction |

In isolation, GPTQ's own space is:

$$
3 \times 5 \times 2 \times 2 \times 3 = 180 \text{ configurations}
$$

But GPTQ isn't run in isolation here — it's chained directly after `SelectiveMixedPrecision`, and the two passes are not independent. `SelectiveMixedPrecision` writes its chosen precision settings into a ``mixed_precision_info`` model attribute, and GPTQ reads that attribute and **overrides its own matching parameters with whatever `SelectiveMixedPrecision` picked**, whenever they differ (see ``get_quant_config`` in ``quant_utils.py``). Once `SelectiveMixedPrecision` has chosen a value for a shared parameter, GPTQ's own sampled value for that same parameter is discarded and never actually takes effect — so multiplying it in twice would double-count the exact same choice.

Crucially, *which* parameters are shared depends on which `algorithm` `SelectiveMixedPrecision` used:

| `SelectiveMixedPrecision` algorithm | Parameters handed down to GPTQ | GPTQ's remaining independent parameters | GPTQ-unique multiplier |
|---|---|---|---|
| Heuristic — ``high_precision_lm_head``, ``high_precision_mlp_down``, ``high_precision_mlp_down_qkv`` | ``bits`` only | ``group_size``, ``sym``, ``lm_head``, ``damp_percent`` | $5 \times 2 \times 2 \times 3 = 60$ |
| Score-based — ``snr``, ``snr_relative``, ``iqe``, ``iqe_relative``, ``kld_gradient`` | ``bits``, ``group_size``, ``sym`` | ``lm_head``, ``damp_percent`` | $2 \times 3 = 6$ |

So the GPTQ-unique multiplier isn't a single number — it ranges from a **minimum of 6** (whenever `SelectiveMixedPrecision` uses a score-based algorithm, e.g. ``kld_gradient``, which hands down all three of ``bits``/``group_size``/``sym``) up to a **maximum of 60** (whenever it uses a heuristic algorithm, e.g. ``high_precision_mlp_down_qkv``, which only hands down ``bits``).

This is useful intuition, but it is not the same thing as the search space Olive actually traverses. In Olive, each pass contributes a `SearchSpace` to the search strategy (`SearchStrategy.initialize` combines them pass-by-pass or jointly), and the mixed-precision override is only applied when a search point is executed (`get_quant_config` in ``quant_utils.py``). In other words, the value above is a **theoretical deduplicated-output count**: a rough upper bound on how many distinct end configurations could result after shared GPTQ settings are overridden at runtime.

The actual declared search space still includes the shared GPTQ choices that are sampled for each pass, and the heuristic-only knobs are not a meaningful reduction in that space: `group_size`/`sym` are unused for heuristic algorithms, and score-based algorithms still require a valid `ratio`. So the correct mental model is:

- `SelectiveMixedPrecision` contributes its declared search parameters, including `algorithm`, `bits`, `group_size`, `sym`, `high_bits`, `high_group_size`, `high_sym`, and `ratio` when left unset.
- GPTQ contributes its own sampled values for the parameters that remain independent after the override.
- The `6,300` number is at most a theoretical count of output-equivalent recipes after applying the runtime override, not the actual size of the search space Olive declares and traverses.

This is why the doc examples keep those extra fields pinned: otherwise the search space silently balloons, and score-based algorithms can even become invalid because `ratio` is required but unset. The broader point still stands: adding *genuinely independent* categorical parameters multiplies the search space, and every extra pass or parameter you leave searchable adds cost. The job of search is to sample that space intelligently; it is not to pretend the runtime override has already reduced it to a smaller, fully deduplicated search budget.

Exhaustively evaluating thousands of configurations — each requiring a full quantization pass and an evaluation run — is still not realistic. This is the actual problem Olive's search strategy exists to solve.

---

## Step 3: Let Olive search smart, not exhaustive

Olive separates *what* the search space looks like from *how* it's explored. You control the latter with ``search_strategy`` in your workflow config:

```json
{
  "search_strategy": {
    "execution_order": "joint",
    "sampler": "tpe",
    "max_samples": 30,
    "seed": 0
  }
}
```

Key levers:

- **`sampler`** — how points are chosen from the space:
  - `sequential` — walks the space in order (exhaustive if you let it run to completion).
  - `random` — samples points uniformly at random.
  - `tpe` — a Tree-structured Parzen Estimator (via Optuna) that uses prior evaluation results to bias future samples toward promising regions instead of guessing blindly.
- **``max_samples``** — hard cap on the number of configurations actually evaluated. With ``tpe``, 30 well-chosen samples out of 6,300 effective configurations can get you very close to the best one — without the cost of trying all 6,300.
- **``execution_order``** — ``joint`` vs. ``pass-by-pass``:
  - ``joint`` searches all passes together in one combined space and can find interactions between passes that a staged search would miss, at the cost of a larger search space.
  - ``pass-by-pass`` is implemented as a nested walk: Olive evaluates the first pass-space, picks the best parent result, then explores the next pass-space under that parent, and can step back to another parent when needed (`SearchStrategy._initialize_step`, `_step_down`, and `_step_up` in ``search_strategy.py``). That means the staged walk still has a worst-case multiplicative structure across the pass spaces, not a simple additive total like `240 + 6` or `240 + 60`. It is a more structured, parent/child traversal of the same pass-space product, which often reduces the search burden in practice but does not turn the problem into an exact sum of independent stage budgets.
- **``stop_when_goals_met``** — stop early once a target metric (e.g. accuracy or size) is satisfied, rather than spending the full ``max_samples`` budget.

A full workflow config chaining the two passes looks like this:

```json
{
  "input_model": { "type": "HfModel", "model_path": "meta-llama/Llama-3.2-1B-Instruct" },
  "systems": {
    "local_system": { "type": "LocalSystem", "accelerators": [{ "device": "gpu" }] }
  },
  "data_configs": [
    {
        "name": "calib_data",
        "type": "HuggingfaceContainer",
        "load_dataset_config": {
            "data_name": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train"
        }
    }
  ],
  "passes": {
    "mixed_precision": {
      "type": "SelectiveMixedPrecision",
      "high_bits": 8,
      "high_group_size": null,
      "high_sym": null,
      "ratio": 0.8
    },
    "gptq": { "type": "Gptq", "data_config": "calib_data" }
  },
  "search_strategy": {
    "execution_order": "joint",
    "sampler": "tpe",
    "max_samples": 30,
    "seed": 0
  },
  "evaluators": {
    "accuracy_and_size_evaluator": {
      "metrics": [
        {
          "name": "accuracy",
          "type": "accuracy",
          "data_config": "calib_data",
          "sub_types": [
            {
              "name": "accuracy_score",
              "priority": 1,
              "goal": { "type": "max-degradation", "value": 0.01 }
            }
          ]
        },
        {
          "name": "size_on_disk",
          "type": "size_on_disk",
          "sub_types": [
            {
              "name": "bytes",
              "priority": 2,
              "higher_is_better": false,
              "goal": { "type": "percent-min-improvement", "value": 20 }
            }
          ]
        }
      ]
    }
  },
  "evaluator": "accuracy_and_size_evaluator",
  "host": "local_system",
  "target": "local_system"
}
```

Note that neither pass specifies ``bits``, ``group_size``, ``sym``, etc. — Olive uses each parameter's built-in ``search_defaults`` automatically. Run it with:

```bash
olive run --config search_config.json
```

Olive evaluates up to 30 sampled configurations (out of the 6,300-point effective joint space),
scores each against your evaluator, and reports the best search point — the specific combination
of ``algorithm``, ``bits``, ``group_size``, ``sym`` (from ``SelectiveMixedPrecision``), plus
``lm_head`` and ``damp_percent`` (from GPTQ) that gave the best trade-off between accuracy and size.

```{caution}
This "neither pass specifies ``bits``, ``group_size``, ``sym``, etc." convenience is also the
most common way a search space silently balloons. Once ``search_strategy`` is enabled, **any pass
parameter you leave unset falls back to its ``search_defaults``** (if it has one) instead of its
plain default value — so a config you thought was mostly fixed can quietly become fully
searchable, and every pass you add multiplies that risk. Before launching a long-running search,
always check the line Olive logs at the start of a search run:

    Search space contains %d search points ...

(emitted from ``Engine._run_search`` in ``engine.py``). If that number is dramatically larger
than you expected, some parameter you meant to pin down is being searched instead — go back and
set it explicitly before committing to a full run.
```

```{tip}
The [quantization sweep across five SLMs](quant-slms.md) is a real-world example of exactly this
kind of search — it shows how block size, symmetry, and mixed-precision settings interact
differently per model family.
```

---

## An alternative for very large models: search one parameter at a time

`tpe` sampling with a `max_samples` budget works well when each trial is cheap enough to run a few dozen of them. But for a very large model, a single GPTQ trial can mean hours of calibration and quantization, plus an evaluation run on top — 30 samples of a 6,300-point space is no longer a quick experiment, it's a multi-day job. In that regime, it's often more practical to abandon automatic joint/pass-by-pass search altogether and instead search **one parameter at a time**, by hand, in a deliberate order.

The idea is a greedy, coordinate-wise sweep: pick the parameter you expect to matter most, sweep only that one while holding everything else fixed at a reasonable default, lock in the winner, then move to the next parameter. Concretely, for ``SelectiveMixedPrecision`` + GPTQ on a very large model, a sensible order is:

1. **``algorithm`` first.** This is the parameter most likely to make or break your accuracy/size trade-off, and it's cheap to compare because you fix everything else (e.g. ``bits``=4, ``group_size``=128, ``sym``=False) and run all 8 candidates:

   ```json
   {
     "type": "SelectiveMixedPrecision",
     "algorithm": [
        "high_precision_lm_head",
        "high_precision_mlp_down",
        "high_precision_mlp_down_qkv",
        "snr",
        "snr_relative",
        "iqe",
        "iqe_relative",
        "kld_gradient"
     ],
     "bits": 4,
     "group_size": 128,
     "sym": false,
     "high_bits": 8,
     "high_group_size": null,
     "high_sym": null,
     "ratio": 0.8
   }
   ```

   8 trials, and you're done with ``algorithm`` for good — say the winner is ``high_precision_mlp_down_qkv``. (Olive treats a plain list assigned to a parameter that has ``search_defaults`` as shorthand for "search over these values" — it's automatically turned into a ``Categorical`` search parameter, so you don't need to spell out the longer ``{"olive_parameter_type": "SearchParameter", ...}`` form by hand.)

2. **``bits`` next, with ``algorithm`` locked to the winner.** Now sweep the 3 candidate bit widths with ``algorithm`` fixed:

   ```json
   {
     "type": "SelectiveMixedPrecision",
     "algorithm": "high_precision_mlp_down_qkv",
     "bits": [2, 4, 8],
     "group_size": 128,
     "sym": false,
     "high_bits": 8,
     "high_group_size": null,
     "high_sym": null,
     "ratio": 0.8
   }
   ```

   3 more trials. Say `4` wins.

3. **``group_size`` last, with ``algorithm`` and ``bits`` both locked.** Sweep the 5 candidate group sizes:

   ```json
   {
     "type": "SelectiveMixedPrecision",
     "algorithm": "high_precision_mlp_down_qkv",
     "bits": 4,
     "group_size": [-1, 16, 32, 64, 128],
     "sym": false,
     "high_bits": 8,
     "high_group_size": null,
     "high_sym": null,
     "ratio": 0.8
   }
   ```

   5 more trials (fewer still if the locked-in `algorithm` is heuristic-based, since `group_size` doesn't affect its output at all and can be skipped).

4. Repeat the same pattern for whatever remains independent on the GPTQ side — ``damp_percent`` (3 trials), and ``lm_head`` (2 trials) if you care about it — each time locking in everything decided so far.

Total: roughly 8 + 3 + 5 + 3 + 2 = **21 trials**, run in a strict sequence, instead of 30 (still fairly cheap) or 6,300 (exhaustive) trials explored jointly. For a model where each trial takes hours, that difference is the difference between finishing in a day versus not finishing at all.

This is a deliberate trade-off, not a free lunch:

- It assumes the parameters are reasonably **separable** — that the best ``bits`` value doesn't secretly depend on which ``group_size`` you'll pick later. When parameters genuinely interact, a greedy sweep can lock in a locally-good but globally-suboptimal combination that a joint search would have found. Order matters: sweep the parameter you have the strongest prior about (or that has the largest expected impact) first, so an early, confident decision doesn't box out a better joint combination.
- It trades search quality for wall-clock time and predictability — you know exactly how many trials you're committing to (21, not "however many ``max_samples`` decides"), which matters when every trial is expensive.
- It's a manual process today: Olive's ``search_strategy`` doesn't have a built-in "one parameter at a time" execution order, so each step above is its own small ``olive run`` invocation (or a ``run_pass`` call) with the just-decided parameters hardcoded and only the next parameter left as a search space.

In practice, this hands-on approach is most valuable exactly where automatic search struggles most: huge models where a single quantize-and-evaluate cycle is the bottleneck, and where you have enough domain intuition to guess a reasonable sweep order.

---

## Another angle: use a smaller calibration and eval dataset during search

Even with TPE sampling or one-parameter-at-a-time sweeps, each trial still requires quantizing the entire model and running the full evaluation suite. For a 70B-parameter model on limited GPU memory, that's hours per trial. A pragmatic way to speed this up is to **use a smaller dataset for calibration and evaluation during the search phase**, then re-run the winner(s) on the full dataset once you've narrowed down the candidates.

The logic is simple: most bad configurations fail obviously on a fraction of the data. A quantization setting that tanks accuracy on the first 5% of a calibration dataset will likely tank it on the full 100% too — no need to spend 12 hours evaluating the complete run. Once you've used this quick screening to eliminate the worst options, you can afford to spend the time validating the top survivors on the full dataset.

Concretely:

1. **Initial search with a small calibration set.** Modify your ``data_config`` to use only 5–10% of your calibration data (e.g. first N samples, or a random subsample). Run your full search (or one-parameter-at-a-time sweep) with this reduced set. This might drop each trial from 4 hours to 30 minutes.

2. **Trim based on preliminary results.** After the initial sweep, look at the top N candidates (e.g. top 3–5). Discard anything that was obviously underperforming.

3. **Re-validate winners on full data.** Run those top candidates again with your full calibration and evaluation dataset to get real accuracy/latency numbers. This is still expensive, but you're only doing it 3–5 times instead of 30 or 300.

This two-stage approach is a form of **coarse-then-fine search**. It's particularly effective when:

- Your model is huge and each quantize-and-evaluate cycle is measured in hours.
- You have a clear "obvious failure" regime (e.g. 2-bit quantization with certain settings always drops accuracy by 20%+) that appears at small scale.
- Your evaluation metric is relatively stable across dataset subsets (i.e. a configuration that's good on 5% of the data is likely good on 100%, just with more precise numbers).

The trade-off is that you're not evaluating the full search space rigorously — a configuration might rank in the middle on 5% of data but top on 100%. But in practice, for quantization, this is rare; the ranking tends to be fairly consistent.

To implement this in Olive, the practical knobs are usually not a generic ``load_dataset_config`` sample cap. ``huggingface_dataset`` forwards unknown kwargs to ``datasets.load_dataset``, so there is no universal ``max_samples`` on that layer; for text-generation workloads the relevant limit is usually in the preprocessing pipeline (for example ``pre_process_data_config.max_samples`` or a component-specific dataset limit). Also, ``kld_gradient`` builds its own default calibration dataset internally, so changing GPTQ's data config does not reduce that scoring cost. Olive doesn't have a built-in "two-stage search" mode, so you'd run two separate ``olive run`` invocations: one with a reduced dataset and an open search space, then a second (no-search, fixed-config) pass with the full dataset and the winning parameters hardcoded.

---

## Practical guidance for keeping the search sane

- **Start narrow.** Override ``search_defaults`` with a smaller candidate list for parameters you already have intuition about (e.g. only ``group_size``: [32, 128]) before opening up the full range.
- **Check the logged search space size before committing to a full run.** Olive logs ``Search space contains %d search points ...`` right before a search starts — if that count is much bigger than you expected, a parameter you meant to fix is silently being searched via its ``search_defaults`` instead.
- **Watch for parameters shared across chained passes.** If a downstream pass reads state written by an upstream pass (as GPTQ reads ``SelectiveMixedPrecision``'s ``mixed_precision_info``), don't count that parameter's options twice — and consider fixing it on the downstream pass entirely so search budget isn't wasted on trials that produce an identical output model.
- **Prefer ``tpe`` over ``sequential``/``random`` once the space exceeds a few hundred points.** TPE's whole purpose is to avoid needing to touch every point.
- **Use ``pass-by-pass`` when passes are largely independent**, and reserve ``joint`` search for passes you suspect interact (like a mixed-precision annotation feeding directly into the quantizer that consumes it).
- **For very large models, run an initial search with a smaller calibration/eval dataset (5–10% of full size) to eliminate obvious failures, then re-validate the top candidates on the full dataset.** This "coarse-then-fine" approach can cut iteration time dramatically while still finding good candidates.
- **For very large models where each trial is expensive, consider a manual one-parameter-at-a-time sweep** instead of automatic search — lock in the parameter you have the strongest prior about first (usually ``algorithm``), then work down the list.
- **Set ``max_samples`` or ``max_time``** so a search has a hard budget instead of running indefinitely.
- **Every new, genuinely independent parameter you add to a joint search is a multiplier, not an addend** — budget your ``max_samples`` accordingly.

---

## What about ONNX?

Everything above happens entirely on the PyTorch model — ``SelectiveMixedPrecision`` and ``Gptq`` are PyTorch-side passes, and the search finds the best PyTorch quantization recipe. Once you have that best search point, converting the winning model to ONNX Runtime is a separate, deterministic step (no search needed): run it through Olive's export passes, e.g. ``capture-onnx-graph`` (or the ``ModelBuilder``/``DynamoExporter`` passes it wraps) to produce the ONNX graph, followed by ``OnnxGraphSurgery`` or ``auto-opt`` to apply ONNX-level graph optimizations. See the [CLI how-to guides](../how-to/cli/cli-optimize.md) for the exact commands.

---

## Summary

Olive's search turns "which quantization settings should I use?" from a guessing game into a structured optimization problem. The catch is that the search space grows multiplicatively with every *genuinely independent* parameter and pass you chain together — a few categorical options per pass can snowball into thousands of configurations once passes are searched jointly. Parameters that are shared between chained passes (like ``bits``, ``group_size``, and ``sym`` between ``SelectiveMixedPrecision`` and GPTQ) shouldn't be double-counted, but every parameter that remains independent still multiplies the space it's added to. Understanding that growth, and using samplers, sample budgets, and execution order deliberately, is what makes searching over ``SelectiveMixedPrecision`` and GPTQ (or any other pass combination) tractable instead of overwhelming.
