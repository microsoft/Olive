# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Manual verification helper: compare wikitext perplexity before/after quantizing a real HF model.

This is a local, one-off validation tool -- it is NOT wired into any Olive workflow config or
CI. It exists to answer "does quantizing this real (downloaded) checkpoint produce a sane
quality regression, or does it blow up / silently corrupt the model?" -- the same class of
question that surfaced real bugs in #2584 (RTN) after synthetic-model unit tests had already
passed.

Usage:
    python scripts/quantize_and_compare_perplexity.py \
        --model_id ibm-granite/granite-3.0-1b-a400m-base \
        --pass_name Gptq \
        --pass_config '{"bits": 4, "group_size": 128, "sym": true, "moe": true}'

By default this evaluates perplexity over the *entire* wikitext-2 test split (use
``--num_samples`` to restrict to a prefix for a quicker/dirtier check). The model can be any HF
model id or local path; the pass can be any registered Olive pytorch *quantization* pass (Gptq,
Rtn, KQuant, AutoAWQQuantizer, GptqQuantizer, GptqModel, ...) with any config -- nothing here is
hardcoded to GPTQ or to MoE. Non-quantization passes (e.g. SparseGPT, which prunes rather than
quantizes) can technically be loaded too, but the size/perplexity comparison this script prints is
framed around quantization and may be misleading for a pruning pass.

Fairness notes (read before trusting a delta from this script):
  * Baseline and quantized models are loaded with the *same* ``--dtype`` (default "auto", i.e.
    each checkpoint's native dtype) and the same ``--experts_implementation``, so a measured
    perplexity delta is attributable to quantization and not to an incidental dtype/backend
    mismatch between the two loads.
  * "Weights size" reports two *different* metrics, both labeled explicitly in the summary: an
    in-memory figure (parameter count x element size at the loaded dtype), computed identically
    for baseline and quantized so that pair is directly comparable, and a separately-labeled
    on-disk figure (actual saved weight-file bytes, quantized side only) that reflects real
    storage compression. Do not compare the in-memory baseline number against the on-disk
    quantized number -- they measure different things.
  * Calibration sample/token counts are captured by intercepting the *actual* dataset the pass
    builds internally (not recomputed separately by this script), so they are structurally
    guaranteed to match what was really used to calibrate -- and are skipped/reported as n/a for
    data-free passes (e.g. RTN) that don't consume a data_config at all.
  * "Quantization time" is wall-clock for the whole ``pass.run()`` call (load + calibrate +
    quantize + save), not a pure inference/perf benchmark -- a data-free pass like RTN being much
    faster than a calibration-based pass like GPTQ is an expected algorithmic trade-off, not a
    regression.
  * ``--device`` controls where the baseline model loads and where perplexity is evaluated for
    both models. It does NOT control where quantization/calibration itself runs -- some passes
    (e.g. Gptq's layerwise calibration) pick their own device internally (cuda when available,
    else cpu) independent of this flag. On a single-GPU machine this is usually moot; on a
    multi-GPU machine, quantization may run on a different GPU than the one named here.
  * A single run against one (possibly non-random) slice of one dataset, on one machine/GPU, is a
    smoke test, not a statistically rigorous benchmark -- treat deltas smaller than the
    run-to-run/sample-to-sample noise floor with caution, especially for small ``--num_samples``
    overrides, and do not treat cited timing numbers as reproducible to more than roughly
    plus-or-minus 20% run-to-run without re-measuring on your own hardware.
  * Calibration data defaults to Olive's own built-in default (wikitext-2 train), matching what a
    real user gets out of the box -- this is what the primary results table should be based on.
    Wikitext-2 train and the (also wikitext-2, by default) eval split are disjoint but same-domain,
    so as a *secondary*, opt-in cross-domain sanity check (not the headline number), evaluate on
    C4 instead: ``--dataset allenai/c4 --dataset_config en --split validation --eval_streaming
    --num_samples 200`` (C4 is sharded across 1000+ files, so non-streaming slicing is
    prohibitively slow -- ``--eval_streaming`` is required for it).
"""

import argparse
import gc
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import torch

# ruff: noqa: T201  # this is a CLI tool; print() output is the point

_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def weights_size_gb(path: Path) -> float:
    """On-disk size of only the model-weight files under ``path``, in GiB.

    Excludes config/tokenizer/vocab/metadata files so this is comparable across checkpoints that
    may differ in how much non-weight metadata they carry.
    """
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and f.suffix in _WEIGHT_FILE_SUFFIXES) / (
        1024**3
    )


def param_memory_gb(model: torch.nn.Module) -> float:
    """In-memory footprint of a model's parameters at their current dtype, in GiB."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)


def capture_moe_fallback_counts() -> tuple[list, callable]:
    """Monkeypatch ``CoverageReport.add`` to capture per-layer MoE fallback (RTN) stats.

    ``Gptq``'s MoE calibration only *logs* its coverage/fallback report -- it doesn't return it
    to the caller. Since the fallback experts (too few calibration tokens for a useful Hessian)
    are exactly the "which experts did NOT get real GPTQ treatment" signal this script wants,
    intercept the structured ``LayerCoverage`` objects before they're formatted into a log
    string, instead of parsing log text.

    Returns (captured_layers, restore_fn). Stays empty for non-MoE passes: the module-level patch
    is applied unconditionally (import is cheap and side-effect-free until ``CoverageReport.add``
    is actually called), but data-free/non-MoE passes never call it, so ``captured`` stays empty.
    """
    from olive.passes.pytorch.moe_calib import CoverageReport

    captured = []
    original_add = CoverageReport.add

    def patched_add(self, coverage):
        captured.append(coverage)
        return original_add(self, coverage)

    CoverageReport.add = patched_add

    def restore():
        CoverageReport.add = original_add

    return captured, restore


def capture_calibration_dataset() -> tuple[dict, callable]:
    """Monkeypatch the calibration-dataset builder actually called by ``run_layerwise_quantization``.

    ``olive.passes.pytorch.quant_utils`` does ``from .train_utils import get_calibration_dataset``,
    binding its own module-level name -- patching ``train_utils.get_calibration_dataset`` would
    not affect that call site, so this patches ``quant_utils.get_calibration_dataset`` directly.
    This captures the *exact* dataset the pass consumes (not a separately-recomputed copy), so
    the reported sample/token counts are structurally guaranteed to match, and for data-free
    passes (e.g. RTN) that never call this function, ``captured`` stays empty.
    """
    import olive.passes.pytorch.quant_utils as quant_utils_mod

    captured: dict = {}
    original = quant_utils_mod.get_calibration_dataset

    def patched(model, data_config):
        dataset = original(model, data_config)
        captured["dataset"] = dataset
        return dataset

    quant_utils_mod.get_calibration_dataset = patched

    def restore():
        quant_utils_mod.get_calibration_dataset = original

    return captured, restore


def compute_perplexity(
    model, tokenizer, text: str, device: str, stride: int = 512, max_len: int = 2048
) -> tuple[float, int]:
    """Sliding-window perplexity over ``text``, per the standard HF perplexity recipe.

    Uses overlapping windows (``max_len`` context, ``stride`` step) so every scored token has a
    long enough context, without requiring the whole document to fit in one forward pass.
    ``max_len`` is a fixed evaluation-window size (matching the common HF perplexity recipe,
    which typically also uses a fixed window rather than each model's full context length) --
    it is applied identically to baseline and quantized so the comparison stays fair even though
    it may be smaller than a given model's actual ``max_position_embeddings``. Override via
    ``--max_len`` if you need a specific window size for your model.

    Returns ``(perplexity, n_scored_tokens)``. ``n_scored_tokens`` is the number of positions
    that actually contributed to the loss -- not ``trg_len``, since a causal-LM's internal
    label-shift means the first position of every window's target has no prediction to score.
    """
    model.eval()
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    seq_len = input_ids.size(1)
    if seq_len < 2:
        raise ValueError(
            f"Eval text tokenized to only {seq_len} token(s) -- need at least 2 for a scoreable "
            "causal-LM window. Check --dataset/--dataset_config/--split/--num_samples."
        )

    nlls = []
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore_index: don't double-count the overlapped prefix
        with torch.no_grad():
            loss = model(ids, labels=target_ids).loss
        # the model shifts labels internally (logits[:-1] vs labels[1:]), so one more position
        # than "trg_len" is unscored; weight by what was actually scored, not the raw window size.
        n_valid = (target_ids[:, 1:] != -100).sum().item()
        nlls.append(loss * n_valid)
        n_tokens += n_valid
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / n_tokens).item(), n_tokens


def load_eval_text(
    dataset: str, dataset_config: str, split: str, num_samples: int | None, streaming: bool = False
) -> str:
    """Load and concatenate non-empty rows of ``split`` for perplexity evaluation.

    ``num_samples=None`` (the default) uses the *entire* split -- wikitext-2's test split has
    many blank/heading-only rows, so a small fixed prefix (e.g. the first 200 rows) can end up
    representing only a small, non-random, unrepresentative slice of the actual text.

    ``streaming=True`` is required for datasets sharded across many files (e.g. ``allenai/c4``,
    used as an optional cross-domain eval set): a non-streaming row-slice like ``split[:50]``
    still triggers enumerating/downloading every shard's metadata to compute slice boundaries,
    which is far slower than just streaming the first ``num_samples`` rows directly. Streaming
    mode requires ``num_samples`` to be set (no notion of "the entire split" for an unbounded
    stream).
    """
    from datasets import load_dataset

    if streaming:
        if num_samples is None:
            raise ValueError("--num_samples is required when using a streaming eval dataset (e.g. C4).")
        ds = load_dataset(dataset, dataset_config, split=split, streaming=True)
        rows = []
        for row in ds:
            if row["text"].strip():
                rows.append(row["text"])
            if len(rows) >= num_samples:
                break
        return "\n\n".join(rows)

    ds = load_dataset(dataset, dataset_config, split=split)
    rows = [row for row in ds["text"] if row.strip()]
    if num_samples is not None:
        rows = rows[:num_samples]
    return "\n\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", required=True, help="HF model id or local model path.")
    parser.add_argument("--pass_name", default="Gptq", help="Olive pytorch pass class name, e.g. Gptq, Rtn, KQuant.")
    parser.add_argument(
        "--pass_config",
        default='{"bits": 4, "group_size": 128, "sym": true}',
        help="JSON dict of pass config kwargs (e.g. bits/group_size/sym/moe/...). Pass an explicit "
        "'data_config' key here to override the calibration dataset (see --calib_* flags for a shortcut).",
    )
    parser.add_argument("--dataset", default="wikitext", help="HF dataset id for the perplexity eval text.")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1", help="HF dataset config name.")
    parser.add_argument("--split", default="test", help="Dataset split to use for the eval text.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Restrict the eval split to the first N (non-empty) rows. Default: use the entire split. "
        "Required (must be set) when --eval_streaming is used.",
    )
    parser.add_argument(
        "--eval_streaming",
        action="store_true",
        help="Stream the eval dataset instead of a plain row-slice load. Required for datasets sharded across "
        "many files, e.g. '--dataset allenai/c4 --dataset_config en --split validation --eval_streaming "
        "--num_samples 200' to use C4 as a cross-domain eval set (checking whether same-domain "
        "wikitext-calibration-on-wikitext-eval inflates quality vs. a genuinely held-out domain).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="torch_dtype used to load BOTH the baseline and the (pre-quantization) input model, so a measured "
        "perplexity delta is attributable to quantization and not to a dtype mismatch between the two loads. "
        "'auto' (default) preserves each checkpoint's native dtype, matching Olive's own default load behavior.",
    )
    parser.add_argument(
        "--calib_dataset",
        default=None,
        help="HF dataset id for calibration data, e.g. 'allenai/c4' for a cross-domain calibration set (the "
        "convention used by the GPTQ/AWQ papers) instead of Olive's wikitext-2 default, which shares a domain "
        "with the default --dataset/--dataset_config eval text above (train/test are disjoint, but same-domain "
        "calibration can still inflate quality on a same-domain eval set relative to true generalization). "
        "Default: use Olive's built-in default (Salesforce/wikitext, wikitext-2-raw-v1). Ignored if --pass_config "
        "already specifies 'data_config', or if the target pass doesn't declare a data_config parameter at all "
        "(e.g. Rtn, which is data-free).",
    )
    parser.add_argument("--calib_dataset_config", default=None, help="HF dataset config name for --calib_dataset.")
    parser.add_argument(
        "--calib_split",
        default=None,
        help="Dataset split to use for calibration data. Default: None, meaning don't override -- inherit "
        "whatever Olive's own current default is (get_calibration_data_config's own 'split' default), so this "
        "script always reflects real Olive behavior even if that default changes in the future, instead of "
        "silently drifting out of sync with a value hardcoded here.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--experts_implementation",
        default="eager",
        help="Set via model.set_experts_implementation(...) on BOTH the baseline and the quantized model before "
        "evaluating a MoE model, so the two loads use the same MoE compute backend. Quantized MoE weights are "
        "storage-only QuantTensors that only the eager experts loop can consume.",
    )
    parser.add_argument(
        "--keep_output",
        action="store_true",
        help="Keep the quantized model directory instead of deleting it when the script exits.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2048,
        help="Sliding-window context length used for perplexity evaluation (applied identically to "
        "baseline and quantized). Default 2048, matching the common HF perplexity recipe -- override "
        "if you specifically need a window matching a given model's max_position_embeddings.",
    )
    args = parser.parse_args()

    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num_samples must be a positive integer.")

    try:
        pass_config = json.loads(args.pass_config)
    except json.JSONDecodeError as e:
        parser.error(f"--pass_config is not valid JSON: {e}")
    if not isinstance(pass_config, dict):
        parser.error("--pass_config must be a JSON object (dict), e.g. '{\"bits\": 4}'.")

    from olive.hardware import DEFAULT_CPU_ACCELERATOR
    from olive.model import HfModelHandler
    from olive.package_config import OlivePackageConfig
    from olive.passes.olive_pass import create_pass_from_dict
    from olive.passes.pytorch.train_utils import get_calibration_data_config

    # Resolve the pass class through Olive's own package registry (olive_config.json) rather than
    # guessing a module path from the class name: several registered passes (e.g. AutoAWQQuantizer
    # in autoawq.py, GptqQuantizer in autogptq.py) do not live in a module named after the
    # lowercased class name, so a naive `olive.passes.pytorch.{name.lower()}` import silently
    # breaks for them.
    pass_cls = OlivePackageConfig.load_default_config().import_pass_module(args.pass_name)
    pass_accepts_data_config = "data_config" in pass_cls.default_config(DEFAULT_CPU_ACCELERATOR)

    print(
        f"=== Loading eval text: {args.dataset}/{args.dataset_config} [{args.split}][:{args.num_samples}] "
        f"(streaming={args.eval_streaming}) ==="
    )
    text = load_eval_text(args.dataset, args.dataset_config, args.split, args.num_samples, args.eval_streaming)

    dtype_kwargs = {} if args.dtype == "none" else {"torch_dtype": args.dtype}

    print(f"=== Loading baseline model: {args.model_id} (dtype={args.dtype}) ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    baseline = AutoModelForCausalLM.from_pretrained(args.model_id, **dtype_kwargs).to(args.device)
    if hasattr(baseline, "set_experts_implementation"):
        baseline.set_experts_implementation(args.experts_implementation)
    baseline_size_gb = param_memory_gb(baseline)
    baseline_dtype = next(baseline.parameters()).dtype

    print("=== Baseline perplexity ===")
    baseline_ppl, baseline_n_tokens = compute_perplexity(baseline, tokenizer, text, args.device, max_len=args.max_len)
    print(f"Baseline perplexity: {baseline_ppl:.4f} (scored {baseline_n_tokens} tokens)")
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    # Load the pre-quantization input model with the SAME dtype kwargs as the baseline above, so the
    # only difference between "baseline" and "quantized" is the quantization itself.
    input_model = HfModelHandler(model_path=args.model_id, load_kwargs=dtype_kwargs)

    if pass_accepts_data_config and "data_config" not in pass_config:
        calib_dataset_kwargs = {}
        if args.calib_split:
            calib_dataset_kwargs["split"] = args.calib_split
        if args.calib_dataset:
            calib_dataset_kwargs["data_name"] = args.calib_dataset
        if args.calib_dataset_config:
            calib_dataset_kwargs["subset"] = args.calib_dataset_config
        data_config = get_calibration_data_config(
            args.model_id,
            trust_remote_code=input_model.get_load_kwargs().get("trust_remote_code", False),
            **calib_dataset_kwargs,
        )
        pass_config = {**pass_config, "data_config": data_config}
    elif not pass_accepts_data_config and (args.calib_dataset or "data_config" in pass_config):
        print(
            f"NOTE: {args.pass_name} does not declare a 'data_config' parameter (data-free quantization); "
            "--calib_dataset / pass_config['data_config'] will be ignored."
        )

    # Capture the calibration dataset actually built and consumed *inside* run_layerwise_quantization
    # (rather than recomputing a separate copy here), so the reported sample/token counts are
    # structurally guaranteed to match what was really used -- and stay empty for data-free passes.
    captured_calib, restore_calib_capture = capture_calibration_dataset()
    fallback_layers, restore_coverage_capture = capture_moe_fallback_counts()

    printable_pass_config = {k: v for k, v in pass_config.items() if k != "data_config"}
    out_dir = Path(tempfile.mkdtemp(prefix="olive_quant_demo_"))
    try:
        print(f"=== Running {args.pass_name} pass with config={printable_pass_config} ===")
        quant_pass = create_pass_from_dict(pass_cls, pass_config, disable_search=True)
        quant_start = time.time()
        output_model = quant_pass.run(input_model, str(out_dir))
        quant_duration_s = time.time() - quant_start
        quantized_weights_size_gb = weights_size_gb(out_dir)

        if "dataset" in captured_calib:
            calib_num_samples = len(captured_calib["dataset"])
            # Sum tokens across the full batch dimension of every row (not just row["input_ids"][0])
            # so this stays correct if a --calib_* override or a future default ever uses batch_size > 1.
            calib_num_tokens = sum(row["input_ids"].numel() for row in captured_calib["dataset"])
            calib_summary = f"{calib_num_samples} samples, {calib_num_tokens} tokens"
        else:
            calib_summary = "n/a (data-free pass)"

        print("=== Loading quantized model ===")
        quantized = output_model.load_model()
        if hasattr(quantized, "set_experts_implementation"):
            quantized.set_experts_implementation(args.experts_implementation)
        quantized = quantized.to(args.device)

        print("=== Quantized perplexity ===")
        quant_ppl, quant_n_tokens = compute_perplexity(quantized, tokenizer, text, args.device, max_len=args.max_len)
        print(f"Quantized perplexity: {quant_ppl:.4f} (scored {quant_n_tokens} tokens)")

        # Apples-to-apples with baseline_size_gb: same measurement method (in-memory param
        # bytes at the loaded dtype) on both sides. weights_size_gb(out_dir) above is a
        # DIFFERENT metric (on-disk artifact size, includes any save-time packing/compression)
        # and is reported separately -- conflating the two was Major finding #3 from review.
        quantized_size_gb = param_memory_gb(quantized)

        total_experts = sum(lc.num_experts for lc in fallback_layers)
        total_starved = sum(lc.starved for lc in fallback_layers)
        total_unseen = sum(lc.unseen for lc in fallback_layers)

        print("\n=== SUMMARY ===")
        print(f"Model:                    {args.model_id}")
        print(f"Pass:                     {args.pass_name}({printable_pass_config})")
        print(f"Dtype (baseline & input): {baseline_dtype} (--dtype={args.dtype})")
        print(f"Experts implementation:   {args.experts_implementation} (applied to both models)")
        print(f"Calibration set:          {calib_summary}")
        print(
            f"Quantization time:        {quant_duration_s:.1f}s "
            "(end-to-end: load + calibrate + quantize + save, NOT a pure inference benchmark)"
        )
        print(f"Baseline weights size:    {baseline_size_gb:.3f} GiB (in-memory params, {baseline_dtype})")
        quantized_dtype = next(quantized.parameters()).dtype
        print(f"Quantized weights size:   {quantized_size_gb:.3f} GiB (in-memory params, {quantized_dtype})")
        print(
            f"Quantized on-disk size:   {quantized_weights_size_gb:.3f} GiB (saved weight files only, different metric)"
        )
        print(
            f"Eval text:                {args.dataset}/{args.dataset_config}[{args.split}]"
            f"{f'[:{args.num_samples}]' if args.num_samples is not None else ' (full split)'}"
        )
        print(f"Baseline perplexity:      {baseline_ppl:.4f} ({baseline_n_tokens} tokens scored)")
        print(f"Quantized perplexity:     {quant_ppl:.4f} ({quant_n_tokens} tokens scored)")
        print(f"Delta:                    {quant_ppl - baseline_ppl:+.4f}")
        if fallback_layers:
            fallback_label = (
                f"{args.pass_name} + RTN fallback"
                if total_starved + total_unseen
                else f"{args.pass_name} (no fallback)"
            )
            print(
                f"MoE fallback experts:     {total_starved + total_unseen}/{total_experts} "
                f"({total_starved} starved, {total_unseen} unseen) across {len(fallback_layers)} MoE layers "
                f"-- quantized with RTN instead of {args.pass_name} due to insufficient calibration coverage "
                f"[label: {fallback_label}]"
            )
            print("MoE per-layer token/expert coverage:")
            for lc in fallback_layers:
                counts = sorted(lc.token_counts)
                print(
                    f"  {lc.layer_name}: {lc.num_experts} experts, tokens/expert "
                    f"min={counts[0]} median={statistics.median(counts):.0f} max={counts[-1]}, "
                    f"{lc.starved} starved, {lc.unseen} unseen"
                )
        else:
            print("MoE fallback experts:     n/a (not a MoE calibration run)")
    finally:
        restore_coverage_capture()
        restore_calib_capture()
        if args.keep_output:
            print(f"Quantized model kept at: {out_dir}")
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
