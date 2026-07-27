# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Isolated worker for the Whisper GenAI speech-generation discrepancy comparison.

``onnxruntime_genai`` runs native code that can hard-crash (segfault) for some Whisper builds,
e.g. on a genai / model-builder version incompatibility. A native crash cannot be caught by a
Python ``try/except`` in the calling process, so the GenAI generation is executed here in a
separate process: if it crashes, the parent simply observes a non-zero exit code and degrades
gracefully instead of taking down the whole optimize workflow.

The module is intentionally self-contained (standard library plus ``onnx`` / ``onnxruntime_genai``
only, no ``olive`` / ``torch`` / ``transformers`` imports) so it can be launched by file path with a
fast, side-effect-free startup.
"""

import copy
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import onnx

logger = logging.getLogger(__name__)


def expand_output_names(template: str, num_layers: int) -> list:
    """Expand a genai_config output-name value into the concrete ONNX output names.

    ONNX Runtime GenAI encodes per-layer cache outputs as ``%d`` templates (e.g.
    ``present_key_cross_%d``) that it expands over the number of layers, while plain outputs
    such as ``logits`` are used verbatim.
    """
    if "%d" in template:
        return [template % layer for layer in range(num_layers)]
    return [template]


def collect_input_templates(model_section: dict) -> set:
    """Collect every declared input name template across all genai_config model sections."""
    templates = set()
    for section in model_section.values():
        if not isinstance(section, dict):
            continue
        inputs = section.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if isinstance(value, str):
                templates.add(value)
    return templates


def reconcile_output_names(genai_config: dict, actual_outputs: dict):
    """Prune Whisper ``genai_config.json`` output names that are absent from the ONNX graphs.

    A genai / model-builder version mismatch can leave the ``model.encoder`` / ``model.decoder``
    ``outputs`` maps referencing tensors that the actual ONNX graph does not produce, which makes
    ``onnxruntime_genai.Model`` fail with ``Invalid output name: ...``. Dropping such unmatched
    entries lets the model load so the discrepancy comparison can still run.

    An output is only pruned when it is *not consumed* elsewhere in the pipeline. Whisper wires each
    KV-cache ``present`` output into the paired ``past`` input of the next step (e.g. the encoder's
    ``present_key_cross_%d`` output feeds the decoder's ``past_key_cross_%d`` input, and the
    decoder's ``present_key_self_%d`` output feeds its own ``past_key_self_%d`` input). Removing a
    consumed output would leave that cache input unwired, so ``onnxruntime_genai`` loads the model
    but then crashes during generation. Such mismatches are left in place so the load fails cleanly
    with ``Invalid output name`` and the caller can skip the comparison instead.

    Args:
        genai_config: Parsed ``genai_config.json`` contents. Not mutated; a reconciled copy is
            returned instead.
        actual_outputs: Mapping of section name (``"encoder"`` / ``"decoder"``) to the set of
            output names actually present in that section's ONNX graph. Sections missing from the
            mapping are left untouched.

    Returns:
        Tuple ``(reconciled_config, pruned)`` where ``pruned`` is a list of
        ``(section, key, template)`` tuples describing each removed entry.

    """
    reconciled = copy.deepcopy(genai_config)
    pruned = []
    model_section = reconciled.get("model")
    if not isinstance(model_section, dict):
        return reconciled, pruned

    consumed_input_templates = collect_input_templates(model_section)

    for section_name, present in actual_outputs.items():
        section = model_section.get(section_name)
        if not isinstance(section, dict):
            continue
        outputs = section.get("outputs")
        if not isinstance(outputs, dict):
            continue
        num_layers = section.get("num_hidden_layers", 0) or 0
        for key in list(outputs):
            value = outputs[key]
            if not isinstance(value, str):
                continue
            expanded = expand_output_names(value, num_layers)
            # Keep the entry when every concrete output name it references actually exists; a plain
            # output with no expansion (e.g. logits) is kept when its single name exists.
            if not expanded or all(name in present for name in expanded):
                continue
            # A ``present`` KV output whose matching ``past`` input is declared is consumed by the
            # decode loop; pruning it would leave that cache input unwired and crash onnxruntime-genai
            # at generation time, so leave it and let the load fail cleanly instead.
            if value.replace("present", "past") in consumed_input_templates:
                continue
            del outputs[key]
            pruned.append((section_name, key, value))

    return reconciled, pruned


def collect_section_outputs(model_dir: Path, genai_config: dict) -> dict:
    """Read the actual ONNX graph output names for each genai_config model section.

    Returns a mapping of section name (``"encoder"`` / ``"decoder"``) to the set of output names
    found in that section's ONNX file. Sections whose ONNX file cannot be read are omitted so the
    reconciliation leaves them untouched.
    """
    actual_outputs = {}
    model_section = genai_config.get("model", {})
    if not isinstance(model_section, dict):
        return actual_outputs
    for section_name in ("encoder", "decoder"):
        section = model_section.get(section_name)
        if not isinstance(section, dict):
            continue
        filename = section.get("filename")
        if not filename:
            continue
        onnx_path = model_dir / filename
        if not onnx_path.is_file():
            continue
        try:
            onnx_model = onnx.load(str(onnx_path), load_external_data=False)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Could not read ONNX outputs from %s: %s", onnx_path, exc)
            continue
        actual_outputs[section_name] = {output.name for output in onnx_model.graph.output}
    return actual_outputs


def load_genai_speech_model(genai_model_path: str):
    """Load an ``onnxruntime_genai.Model`` for a Whisper model, tolerating output-name mismatches.

    A genai / model-builder version incompatibility can leave ``genai_config.json`` declaring ONNX
    outputs that the graph does not produce, which makes ``og.Model`` raise ``Invalid output
    name: ...``. When that happens, the ``genai_config.json`` is reconciled against the real graph
    outputs in a temporary directory (the large ONNX files are hard-linked, falling back to copies)
    and the load is retried once.

    Returns a tuple ``(genai_model, temp_dir)`` where ``temp_dir`` is a
    ``tempfile.TemporaryDirectory`` that the caller must keep alive while the model is in use and
    clean up afterwards (``None`` when no reconciliation was needed).
    """
    import onnxruntime_genai as og

    try:
        return og.Model(genai_model_path), None
    except Exception as exc:  # pylint: disable=broad-except
        if "invalid output name" not in str(exc).lower():
            raise

        model_dir = Path(genai_model_path)
        genai_config_path = model_dir / "genai_config.json"
        if not genai_config_path.is_file():
            raise
        with genai_config_path.open() as f:
            genai_config = json.load(f)

        actual_outputs = collect_section_outputs(model_dir, genai_config)
        reconciled, pruned = reconcile_output_names(genai_config, actual_outputs)
        if not pruned:
            # Nothing safe to reconcile; the failure is unrelated to prunable output names.
            raise

        logger.warning(
            "Reconciling Whisper genai_config.json: %d output name(s) declared but absent from the "
            "ONNX graph were removed so the model can load (%s). This indicates a genai / "
            "model-builder version mismatch.",
            len(pruned),
            ", ".join(f"{section}.{key}={template}" for section, key, template in pruned),
        )

        temp_dir = tempfile.TemporaryDirectory(prefix="olive_genai_reconciled_")  # pylint: disable=consider-using-with
        temp_path = Path(temp_dir.name)
        for item in model_dir.iterdir():
            if item.name == "genai_config.json" or not item.is_file():
                continue
            target = temp_path / item.name
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
        with (temp_path / "genai_config.json").open("w") as f:
            json.dump(reconciled, f, indent=4)

        try:
            return og.Model(str(temp_path)), temp_dir
        except Exception:
            temp_dir.cleanup()
            raise


def whisper_decoder_prompt(genai_model_path: str) -> str:
    """Build the Whisper decoder prompt string, mirroring olive_evaluator's genai speech path."""
    genai_config = {}
    genai_config_path = Path(genai_model_path) / "genai_config.json"
    if genai_config_path.is_file():
        with genai_config_path.open() as f:
            genai_config = json.load(f)
    # English-only Whisper checkpoints (vocab_size=51864) use a shorter prompt without a
    # language/task selector.
    vocab_size = genai_config.get("model", {}).get("vocab_size", 51865)
    if vocab_size == 51864:
        prompt_tokens = ["<|startoftranscript|>", "<|notimestamps|>"]
    else:
        prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    return "".join(prompt_tokens)


def generate(genai_model_path: str, wav_path: str, max_new_tokens: int, first_n: int) -> dict:
    """Run ONNX Runtime GenAI audio transcription and return the decoded tokens and timings.

    Returns a dict with ``genai_tokens`` (list of ints), ``genai_ttft_s`` and ``genai_ttfn_s``.
    """
    import onnxruntime_genai as og

    genai_model, genai_temp_dir = load_genai_speech_model(genai_model_path)
    try:
        genai_processor = genai_model.create_multimodal_processor()
        prompt = whisper_decoder_prompt(genai_model_path)
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        audios = og.Audios.open_bytes(wav_bytes)
        inputs = genai_processor([prompt], audios=audios)

        params = og.GeneratorParams(genai_model)
        params.set_search_options(do_sample=False, max_length=max_new_tokens + 64, min_length=0, batch_size=1)
        og_generator = og.Generator(genai_model, params)
        og_generator.set_inputs(inputs)

        genai_ttft = None
        genai_ttfn = None
        num_generated = 0
        start = time.perf_counter()
        while not og_generator.is_done():
            og_generator.generate_next_token()
            num_generated += 1
            if num_generated == 1:
                genai_ttft = time.perf_counter() - start
            if num_generated == first_n:
                genai_ttfn = time.perf_counter() - start
            if num_generated >= max_new_tokens:
                break
        genai_tokens = list(og_generator.get_sequence(0))
        del og_generator
    finally:
        del genai_model
        if genai_temp_dir is not None:
            genai_temp_dir.cleanup()

    return {
        "genai_tokens": [int(token) for token in genai_tokens],
        "genai_ttft_s": genai_ttft,
        "genai_ttfn_s": genai_ttfn,
    }


def main(argv=None) -> int:
    """Entry point: read a request JSON and write the generation result JSON.

    Usage: ``python _genai_speech_worker.py <request_json_path> <result_json_path>``. The request
    JSON provides ``genai_model_path``, ``wav_path``, ``max_new_tokens`` and ``first_n``.
    """
    logging.basicConfig(level=logging.WARNING)
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        sys.stderr.write("usage: _genai_speech_worker.py <request_json_path> <result_json_path>\n")
        return 2

    request_path, result_path = argv
    with open(request_path) as f:
        request = json.load(f)

    result = generate(
        genai_model_path=request["genai_model_path"],
        wav_path=request["wav_path"],
        max_new_tokens=int(request["max_new_tokens"]),
        first_n=int(request["first_n"]),
    )
    with open(result_path, "w") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
