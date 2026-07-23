 # LMMSEvaluator vs. Olive's Built-in Metric Path

 ## Context
 Olive already supports multimodal evaluation **two** ways:
 - **(A) Built-in metric path** — recipe uses `data_configs` + `metrics` (`exact_match`, `relaxed_accuracy`, `word_sort_ratio`,`wer`). Olive owns data loading, inference (`OnnxEvaluator._inference_vision_genai`), and scoring.
 - **(B) Harness path (`LMMSEvaluator`)** — recipe uses `"type": "LMMSEvaluator"` + `tasks`. lmms-eval owns data, inference (viathe `ortgenai_mm` adapter), and scoring.

 Both run with `olive run --config <recipe>.json`. Dispatch happens at `olive_evaluator.py:2488`: `Registry.get(self.type orstr(model.framework))`.

 ## Advantages of LMMSEvaluator

 ### 1. Official, literature-matching scorers (most important)
 Built-in metrics are **approximations** of the official ones:

 | Benchmark | Built-in path | Official metric (lmms-eval) | Same number? |
 |-----------|---------------|------------------------------|--------------|
 | TextVQA / MMMU | `exact_match` (string equality) | VQA-accuracy: `min(#match/3, 1)` over 10 answers + normalization | No |
 | DocVQA | `word_sort_ratio` (word overlap) | ANLS (normalized edit distance) | No |
 | ChartQA | `relaxed_accuracy` (±5%) | ±5% relaxed acc with more normalization | Close, not identical |

 For a sensitivity table that must be comparable to published work (Jambay's blog, MBQ, Q-VLM), approximate scorers introduce asilent offset.

 ### 2. Benchmark breadth without writing code
 - **Built-in:** ~4 task shapes (`vision-vqa`, `vision-chart-qa`, `vision-ocr`, `wer`). Each new benchmark needs a data_config + possibly a new preprocessor/metric.
 - **LMMSEvaluator:** ~100 prebuilt benchmarks (OCRBench, MMStar, MathVista, SEED, OmniBench, LibriSpeech, CoVoST, AIR-Bench, VoiceBench…) — each is **one string** in a `tasks` list.

 ### 3. Audio / omni support
 Olive's built-in audio support is **narrow and split across two separate paths**, neither of which covers omni multimodal LLMs:
 - The vision benchmark path (`_inference_vision_genai`) is **vision-only** — it hardcodes `images=` and has no audio handling.
 - There **is** a built-in **ASR** path: WER/RTFx metrics (Olive #2444, wired into olive-recipes #398) for *dedicated speech models* (Whisper, Nemotron) via `_inference_text_genai` / `_inference_text_genai_streaming`. This is audio→text transcription scoring only.

 So what has **no** built-in path is **audio/omni-modal benchmark eval on multimodal LLMs** — e.g. running FLEURS/CoVoST/AIR-Bench/VoiceBench on a Gemma-4-class model that takes interleaved image + audio + text. `LMMSEvaluator` handles image **and** audio (and combined) through one adapter against a single model. Validated end-to-end on quantized Gemma-4: vision (OCRBench, DocVQA) and audio (LibriSpeech WER 7.69, FLEURS WER 19.34) ran together in a single `olive run`.

 ### 4. Consistency with the text story
 Olive already wraps lm-eval for text (`LMEvaluator`). `LMMSEvaluator` is the same pattern for multimodal (wraps lmms-eval) — one mental model: "harness evaluator = standard tasks + standard scorers."

 ## Why new code was needed even though the inference engine is the same

 A fair question: if `LMMSEvaluator` runs the model through the **same** ORT-GenAI/onnxruntime forward pass as the built-in path, why write new code at all?

 Because the new code **isn't** the inference engine — it's the **orchestration/adapter layer** that lets the external lmms-eval harness *drive* that engine. The forward pass (`og.Generator` → onnxruntime) is reused; everything that connects a benchmark harness to it is new and did not exist:

 1. **The adapter (`ortgenai_mm`, `olive/evaluator/lmms_ort.py`).** lmms-eval can only evaluate a model that implements *its* interface — a subclass of `lmms` exposing `generate_until` and `loglikelihood`. ORT-GenAI exposes no such object, and Olive's built-in `_inference_vision_genai` is internal Olive code lmms-eval cannot call. The adapter is the translation layer: it maps lmms-eval's per-task requests (prompts, `doc_to_visual` image/audio payloads) onto ORT-GenAI's API (multimodal processor, `apply_chat_template`, `set_inputs`, the token loop, `get_logits` for loglikelihood scoring), and handles prompt/media-token construction and EOS/stop logic.
 2. **The Olive evaluator (`LMMSEvaluator`).** A new evaluator class registered in Olive's `Registry` that resolves the ORT-GenAI package, hands it to lmms-eval's `simple_evaluate`, and converts the harness's results back into Olive's metric format.
 3. **`CompositeToOnnxPackage`.** The quantized model is emitted by MobiusBuilder as a multi-component composite that can't be loaded/evaluated directly; this pass flattens it into a runnable ORT-GenAI package.

 So: **same inference engine, new control-flow + data + scoring layer.** The metric-based path has Olive own the loop and apply approximate scorers; the harness-based path has lmms-eval own the loop and apply official scorers — and bridging Olive's quantized ONNX model into that harness is precisely what the new code provides.

 ## Caveats / Disadvantages
 1. **For vision, a fidelity/convenience layer rather than a brand-new capability** — built-in `_inference_vision_genai` already runs quantized ONNX VLMs through nearly identical ORT-GenAI inference; for vision, `LMMSEvaluator` swaps the task+scorer layer (official scorers, ~100 tasks) on top of the same engine. **For audio/omni-modal LLM benchmarks it *is* a new capability** — no built-in path existed (the built-in ASR path only covers dedicated speech models, not omni LLMs).
 2. **Heavier dependency** — pulls in lmms-eval + transitive deps (`editdistance`, `more_itertools`, `datasets` versionconstraints, `torchcodec`/FFmpeg).
 3. **Less control / more opacity** — inherits lmms-eval's task definitions and quirks; version-coupled to the external harness.

 ## One-sentence summary
 For vision, LMMSEvaluator mostly upgrades an eval path Olive already had (approximate scorers → official ones, ~4 task shapes → ~100 ready benchmarks); for audio/omni-modal LLMs it adds a path Olive genuinely lacked — and in both cases it trades a heavier dependency and less control for **official literature-matching scorers, ~100 ready benchmarks, and image+audio+omni coverage in one adapter**, which is exactly what a citable cross-model quantization-sensitivity sweep needs and what the built-in approximate-scorer path can't provide without substantial per-benchmark coding.