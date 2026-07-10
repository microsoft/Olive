# -------------------------------------------------------------------------
# ORT-GenAI wrapper for lmms-eval multimodal evaluation
#
# Supports evaluating multimodal ONNX
# models through the EvolvingLMMs-Lab/lmms-eval harness, mirroring how
# olive/evaluator/lmeval_ort.py wraps lm-evaluation-harness for text models.
#
# Registers an LMMSORTGenAIEvaluator class with lmms-eval's legacy
# @register_model registry under the name "ortgenai_mm". Consumers obtain it
# via lmms_eval.api.registry.get_model("ortgenai_mm").
# -------------------------------------------------------------------------
"""ORT-GenAI wrapper for lmms-eval multimodal evaluation."""

from __future__ import annotations

import io
import json
import logging
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
from tqdm import tqdm

try:
    import onnxruntime_genai as og
except ImportError:  # pragma: no cover - optional dep
    og = None

try:
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
    from lmms_eval.api.registry import register_model

    _LMMS_EVAL_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - optional dep
    Instance = Any
    _LMMS_EVAL_IMPORT_ERROR = e

    class lmms:  # noqa: N801
        pass

    def register_model(_name):
        def decorator(cls):
            return cls

        return decorator


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def warmup_image_decoder() -> None:
    """Force ORT-GenAI's image decoder to initialize once, early in the process.

    ORT-GenAI's ``ImageDecoder`` statically links its own zlib. Pillow, torch,
    ``decord`` and other libraries loaded by the HuggingFace evaluation path also
    bundle their own zlib. Because these are static copies of the same symbols
    (an ODR violation), the *first* library to actually decode an image binds its
    zlib process-wide. When ``LMMSEvaluator`` runs an HF input-model evaluation
    before the ORT-GenAI (ONNX) evaluation in the same process
    (``evaluate_input_model: true``), the HF path decodes images first, so
    ORT-GenAI's PNG decoder later binds the wrong zlib and fails with
    ``libpng error: bad parameters to zlib`` on every image.

    Decoding a tiny throwaway PNG through ORT-GenAI here — before the HF path
    runs — makes ORT-GenAI's zlib win the binding, so the later ONNX evaluation
    decodes correctly. Idempotent and best-effort: a no-op if onnxruntime-genai
    is unavailable or the warmup fails (nothing is broken for HF-only usage).
    """
    if og is None:
        return
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warmup.png"
            PIL.Image.new("RGB", (8, 8), (0, 0, 0)).save(path, format="PNG")
            # og.Images.open touches ORT-GenAI's libpng/zlib, binding its symbols.
            og.Images.open(str(path))
    except Exception as e:  # pragma: no cover - defensive, must never break a run
        logger.debug("ORT-GenAI image decoder warmup skipped: %s", e)


_PROVIDER_ALIASES = {
    "cuda": "cuda",
    "cudaexecutionprovider": "cuda",
    "gpu": "cuda",
    "cpu": "cpu",
    "cpuexecutionprovider": "cpu",
    "dml": "dml",
    "dmlexecutionprovider": "dml",
    "directml": "dml",
    "webgpu": "webgpu",
    "webgpuexecutionprovider": "webgpu",
    "js": "web",
    "jsexecutionprovider": "web",
    "nvtensorrtrtx": "NvTensorRtRtx",
    "nvtensorrtrtxexecutionprovider": "NvTensorRtRtx",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_MEDIA_PLACEHOLDER_RE = re.compile(r"(<image>|<audio>)")


def _normalize_image(visual) -> PIL.Image.Image | None:
    """Coerce an lmms-eval visual into an RGB PIL image, or None if it isn't an image.

    Accepts PIL images, file paths, ``{"bytes"|"path"}`` dicts, and numpy arrays;
    audio-shaped dicts (with ``array``/``sampling_rate``) are rejected here.
    """
    if isinstance(visual, PIL.Image.Image):
        return visual.convert("RGB")
    if isinstance(visual, (str, Path)):
        p = Path(visual)
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            return PIL.Image.open(p).convert("RGB")
        return None
    if isinstance(visual, dict):
        # Audio dicts typically include "sampling_rate" or "array"; skip those.
        if "sampling_rate" in visual or "array" in visual:
            return None
        if "bytes" in visual:
            return PIL.Image.open(io.BytesIO(visual["bytes"])).convert("RGB")
        if "path" in visual:
            p = Path(visual["path"]) if visual["path"] else None
            if p is not None and p.suffix.lower() in _IMAGE_SUFFIXES:
                return PIL.Image.open(p).convert("RGB")
            return None
    if isinstance(visual, np.ndarray):
        return PIL.Image.fromarray(np.uint8(visual)).convert("RGB")
    return None


def _normalize_audio(visual) -> tuple[np.ndarray, int] | None:
    """Coerce an lmms-eval visual into ``(mono float32 waveform, sample_rate)``, or None.

    Handles the legacy ``{"array", "sampling_rate"}`` dict, file paths, and
    HF datasets 5.x ``torchcodec`` AudioDecoder objects (duck-typed, downmixed to mono).
    """
    if isinstance(visual, dict):
        if "array" in visual and "sampling_rate" in visual:
            return _normalize_audio_array(visual["array"]), int(visual["sampling_rate"])
        if visual.get("path"):
            return _load_audio_file(Path(visual["path"]))
    if isinstance(visual, (str, Path)):
        return _load_audio_file(Path(visual))
    # torchcodec.decoders.AudioDecoder — HF datasets 5.x returns this for the
    # "audio" feature instead of the legacy {"array", "sampling_rate"} dict.
    # Detect by duck-typing the get_all_samples() method to avoid a hard
    # torchcodec import (it's an optional install).
    if hasattr(visual, "get_all_samples"):
        try:
            samples = visual.get_all_samples()
            # samples.data is a torch.Tensor of shape [channels, num_samples].
            # ORT-GenAI's processor wants mono float32; downmix if multichannel.
            return _normalize_audio_array(samples.data.detach().cpu().numpy()), int(samples.sample_rate)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to decode AudioDecoder visual: %s", e)
            return None
    return None


def _normalize_audio_array(array) -> np.ndarray:
    """Return a contiguous mono float32 waveform from common audio layouts."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] <= 8:
            arr = arr.mean(axis=0)
        elif arr.shape[1] <= 8:
            arr = arr.mean(axis=1)
        else:
            raise ValueError(f"Cannot infer channel axis for audio array with shape {arr.shape}.")
    if arr.ndim != 1:
        raise ValueError(f"Audio array must be one- or two-dimensional, got shape {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _load_audio_file(p: Path) -> tuple[np.ndarray, int] | None:
    """Load an audio file to ``(mono float32 waveform, native sample_rate)`` via librosa.

    Returns None if the extension is unsupported, the file is missing, or librosa isn't installed.
    """
    if p.suffix.lower() not in _AUDIO_SUFFIXES or not p.exists():
        return None
    try:
        import librosa
    except ImportError:
        logger.warning("Audio file %s encountered but librosa not installed.", p)
        return None
    arr, sr = librosa.load(str(p), sr=None, mono=True)
    return arr.astype(np.float32), int(sr)


def _partition_visuals(visuals):
    """Split a mixed list of visuals into ``(images, audios)``.

    Audio is tested first (its ``array``+``sampling_rate`` signature is more distinctive
    than the various image forms).
    """
    images, audios = [], []
    for v in visuals or []:
        if v is None:
            continue
        # Try audio first since its signature ("array"+"sampling_rate") is more
        # distinctive than the image path/bytes/PIL signatures.
        au = _normalize_audio(v)
        if au is not None:
            audios.append(au)
            continue
        img = _normalize_image(v)
        if img is not None:
            images.append(img)
            continue
        suffix = Path(v).suffix.lower() if isinstance(v, (str, Path)) else ""
        if suffix in _VIDEO_SUFFIXES:
            raise NotImplementedError("LMMSORTGenAIEvaluator does not support video inputs.")
        raise ValueError(f"Unsupported lmms-eval visual input type: {type(v).__name__}.")
    return images, audios


def _generation_options(gen_kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str] | None]:
    """Translate lmms-eval generation kwargs to ORT-GenAI search and stop options."""
    if gen_kwargs.get("num_beams") not in (None, 1):
        raise NotImplementedError("Beam search is not supported by LMMSORTGenAIEvaluator.")
    if gen_kwargs.get("num_return_sequences") not in (None, 1):
        raise NotImplementedError("Multiple return sequences are not supported by LMMSORTGenAIEvaluator.")
    if gen_kwargs.get("no_repeat_ngram_size") not in (None, 0):
        raise NotImplementedError("no_repeat_ngram_size is not supported by ORT-GenAI.")
    if gen_kwargs.get("max_agentic_steps") is not None or gen_kwargs.get("visual_cot"):
        raise NotImplementedError("Agentic and visual-CoT generation are not supported by LMMSORTGenAIEvaluator.")

    search_options: dict[str, Any] = {"do_sample": bool(gen_kwargs.get("do_sample", False))}
    for name in ("temperature", "top_p", "top_k", "repetition_penalty"):
        value = gen_kwargs.get(name)
        if value is not None:
            search_options[name] = value

    stop = gen_kwargs.get("until")
    if isinstance(stop, str):
        stop = [stop]
    elif stop is not None:
        stop = list(stop)
    extra_stops = gen_kwargs.get("stop_sequences")
    if isinstance(extra_stops, str):
        extra_stops = [extra_stops]
    if extra_stops:
        stop = [*(stop or []), *extra_stops]
    return search_options, stop or None


def _format_media_tokens(num_items: int, token_format: str) -> list[str]:
    """Expand a media-token format string into one marker per item (1-based ``index``, 0-based ``zero_index``)."""
    return [token_format.format(index=i + 1, zero_index=i) for i in range(num_items)]


def _build_prompt(
    model_type: str,
    num_images: int,
    num_audios: int,
    user_text: str,
    system_prompt: str = "You are a helpful AI assistant.",
    prompt_template: str | None = None,
    image_token_format: str = "<|image_{index}|>",
    audio_token_format: str = "<|audio_{index}|>",
    whisper_language: str = "en",
    whisper_task: str = "transcribe",
    whisper_timestamps: bool = False,
) -> str:
    """Build a chat-style prompt for the model.

    Defaults to Phi-4-multimodal's chat template. For Whisper (pure ASR; no
    text prompt, no chat template), returns an empty string — the ORT-GenAI
    multimodal processor builds the decoder start tokens from the audio input
    plus genai_config defaults.

    Other multimodal architectures use different placeholder tags. Users can
    override the media token formats and the full prompt template from the
    Olive evaluator config without changing this adapter.
    """
    if model_type == "whisper":
        if whisper_task not in ("transcribe", "translate"):
            raise ValueError("whisper_task must be 'transcribe' or 'translate'.")
        timestamp_token = "" if whisper_timestamps else "<|notimestamps|>"
        return f"<|startoftranscript|><|{whisper_language}|><|{whisper_task}|>{timestamp_token}"
    image_tokens = "".join(_format_media_tokens(num_images, image_token_format))
    audio_tokens = "".join(_format_media_tokens(num_audios, audio_token_format))
    parts = [image_tokens, audio_tokens, user_text]
    user_content = "".join(parts)
    if prompt_template:
        return prompt_template.format(
            system_prompt=system_prompt,
            user_content=user_content,
            text=user_text,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            model_type=model_type,
        )

    return f"<|system|>{system_prompt}<|end|><|user|>{user_content}<|end|><|assistant|>"


def _normalize_execution_provider(execution_provider: Any | None) -> str:
    """Normalize an EP spec to a canonical ORT-GenAI provider name.

    Accepts a string or list/tuple (first entry used); empty input returns
    ``"follow_config"``, meaning "use the providers already in genai_config.json".
    """
    if not execution_provider:
        return "follow_config"
    if isinstance(execution_provider, (tuple, list)):
        execution_provider = execution_provider[0] if execution_provider else None
    if not execution_provider:
        return "follow_config"
    ep = str(execution_provider).lower().replace("_", "")
    return _PROVIDER_ALIASES.get(ep, str(execution_provider))


# -----------------------------------------------------------------------------
# Main adapter
# -----------------------------------------------------------------------------


@register_model("ortgenai_mm")
class LMMSORTGenAIEvaluator(lmms):
    r"""lmms-eval model wrapper for an ORT-GenAI multimodal package (model).

    Implements the base lmms class interface for generate_until and loglikelihood, using
    ORT-GenAI's multimodal processor and generator. Supports image and audio inputs, with optional structured content prompt building.
    """

    is_simple = True

    def __init__(
        self,
        pretrained: str,
        batch_size: int = 1,
        max_new_tokens: int = 256,
        max_length: int = 32768,
        system_prompt: str | None = None,
        execution_provider: str | None = None,
        provider_options: dict | None = None,
        fail_on_error: bool = True,
        prompt_template: str | None = None,
        image_token_format: str | None = None,
        audio_token_format: str | None = None,
        ignore_stop_strings: list[str] | None = None,
        whisper_language: str = "en",
        whisper_task: str = "transcribe",
        whisper_timestamps: bool = False,
        **kwargs,
    ) -> None:
        if _LMMS_EVAL_IMPORT_ERROR is not None:
            raise ImportError(
                "lmms-eval is required for ortgenai_mm. Install lmms-eval before using LMMSEvaluator."
            ) from _LMMS_EVAL_IMPORT_ERROR
        if og is None:
            raise ImportError(
                "onnxruntime-genai is required for ortgenai_mm. "
                "Install with: pip install onnxruntime-genai (or -cuda variant)."
            )
        super().__init__()
        if kwargs:
            logger.warning("Unused kwargs: %s", kwargs)

        model_dir = Path(pretrained).resolve()
        if not model_dir.is_dir():
            raise ValueError(f"ORT-GenAI model directory does not exist: {model_dir}")
        if not (model_dir / "genai_config.json").is_file():
            raise ValueError(f"LMMSEvaluator requires genai_config.json in ORT-GenAI package: {model_dir}")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        if int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if int(max_length) < 1:
            raise ValueError("max_length must be >= 1")

        self.model_dir = str(model_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.max_length = int(max_length)
        self.batch_size_per_gpu = int(batch_size)
        self.system_prompt = system_prompt
        self.fail_on_error = fail_on_error
        self.prompt_template = prompt_template
        self.image_token_format = image_token_format
        self.audio_token_format = audio_token_format
        self.whisper_language = whisper_language
        self.whisper_task = whisper_task
        self.whisper_timestamps = bool(whisper_timestamps)
        # Stop strings to suppress from each request's lmms-eval ``until`` list.
        # lmms-eval injects ``until=["\n\n"]`` (the fewshot_delimiter) for tasks
        # that don't set their own ``until`` (lmms_eval/api/task.py). For
        # step-by-step reasoning that double-newline truncates the model after its
        # first paragraph, before it reaches the final answer. Listing "\n\n" here
        # drops only that spurious stop; the model still stops normally on its
        # EOS / end-of-turn token (handled in _run_generation).
        self.ignore_stop_strings = set(ignore_stop_strings or [])

        logger.info("Loading ORT-GenAI model from: %s", self.model_dir)
        ep = _normalize_execution_provider(execution_provider)
        # CUDA GenAI packages often carry provider-specific options in genai_config.json.
        # Clearing/re-adding CUDA can drop those options and fail to load on otherwise
        # working packages, so follow the package config unless options are overridden.
        if ep == "follow_config" or (ep == "cuda" and not provider_options):
            self._model = og.Model(self.model_dir)
        else:
            config = og.Config(self.model_dir)
            config.clear_providers()
            if ep != "cpu":
                config.append_provider(ep)
            for key, value in (provider_options or {}).items():
                config.set_provider_option(ep, key, value)
            self._model = og.Model(config)
        self._tokenizer = og.Tokenizer(self._model)
        self._processor = self._model.create_multimodal_processor()

        # Default prompt-builder path: og.Tokenizer.apply_chat_template. Older
        # onnxruntime-genai versions don't expose this method, in which case we
        # fall back to the legacy format-string path (_build_prompt below).
        self._has_chat_template = hasattr(self._tokenizer, "apply_chat_template")
        if not self._has_chat_template:
            logger.warning(
                "ORT-GenAI tokenizer does not expose apply_chat_template; falling back to "
                "legacy format-string prompt building. Consider upgrading onnxruntime-genai."
            )

        eos_ids = self._tokenizer.eos_token_ids
        self._eos_token_ids = {int(t) for t in (eos_ids if eos_ids is not None else [])}

        try:
            cfg = json.loads((Path(self.model_dir) / "genai_config.json").read_text(encoding="utf-8"))
            self._model_type = cfg.get("model", {}).get("type", "phi4mm")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid genai_config.json in {self.model_dir}") from e

        # Resolve the default system prompt per model family (matching lmms-eval's
        # per-model wrappers' default system prompt) when the user did not pass one explicitly. This keeps
        # HF-input and ORT-GenAI evaluation prompts identical
        if self.system_prompt is None:
            self.system_prompt = self.default_system_prompt_for_model_type(self._model_type)
            logger.info("Using default system prompt for model type '%s': %r", self._model_type, self.system_prompt)

        # Probe (once) whether this model's chat template injects media tokens
        # from structured content parts (``{"type": "image"}``). Well-behaved
        # templates (Gemma-4, Qwen2.5-VL, Qwen3-VL) do (pretty much most); Phi-4-MM's template
        # stringifies the content list as Python repr instead. When supported,
        # the adapter lets the template emit the correct per-model media tokens
        # automatically, so the user does not need to set ``image_token_format``.
        self._supports_structured_content = self._probe_structured_content_support()

        self._rank = 0
        self._world_size = 1
        logger.info("Model loaded. Model type: %s", self._model_type)

    # -------------------------------------------------------------------------
    # lmms-eval required properties
    # -------------------------------------------------------------------------
    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # -------------------------------------------------------------------------
    # ORT-GenAI input plumbing
    # -------------------------------------------------------------------------
    def _build_og_images(self, images, tmp_dir: Path):
        """Write PIL images to temp PNGs and open them as an ``og.Images`` handle (None if no images)."""
        if not images:
            return None
        paths = []
        for i, img in enumerate(images):
            path = tmp_dir / f"image_{i}.png"
            img.save(path, format="PNG")
            paths.append(str(path))
        return og.Images.open(*paths)

    def _build_og_audios(self, audios, tmp_dir: Path):
        """Write ``(waveform, sr)`` pairs to temp WAVs and open them as an ``og.Audios`` handle (None if none)."""
        if not audios:
            return None
        import soundfile as sf

        paths = []
        for i, (arr, sr) in enumerate(audios):
            path = tmp_dir / f"audio_{i}.wav"
            sf.write(path, arr, sr, subtype="FLOAT")
            paths.append(str(path))
        return og.Audios.open(*paths)

    def _handle_error(self, message: str, exc: Exception, default):
        """Re-raise as RuntimeError when ``fail_on_error`` is set; otherwise log and return ``default``."""
        if self.fail_on_error:
            raise RuntimeError(message) from exc
        logger.exception("%s", message)
        return default

    # -------------------------------------------------------------------------
    # Single-request inference primitives
    # -------------------------------------------------------------------------
    def _run_generation(
        self,
        prompt: str,
        images,
        audios,
        max_new_tokens: int,
        stop_strings: list[str] | None = None,
        search_options: dict[str, Any] | None = None,
    ) -> str:
        """Greedily decode a single multimodal request and return the generated text.

        Feeds ``prompt`` + media through the ORT-GenAI processor, then decodes
        token-by-token until EOS, the ``max_new_tokens`` budget, or a ``stop_strings``
        match (early-truncated). Returns "" on processor/input failure unless
        ``fail_on_error`` is set.
        """
        params = og.GeneratorParams(self._model)
        # `max_length` is total (prompt + completion). Image prompts can be huge
        # (Phi-4-MM image embeds are 1000+ tokens), so default generously.
        params.set_search_options(max_length=self.max_length, **(search_options or {"do_sample": False}))
        generator = og.Generator(self._model, params)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            og_images = self._build_og_images(images, tmp_dir)
            og_audios = self._build_og_audios(audios, tmp_dir)

            try:
                # ORT-GenAI multimodal processors disagree on argument shape:
                #   - Phi-4-MM expects a bare string. Passing [prompt] raises
                #     "Number of image tokens does not match the number of images"
                #     because the processor interprets the list as one prompt per
                #     image (verified against pre-built phi4mm INT4 package).
                #   - Whisper's processor (per ORT-GenAI's reference
                #     benchmark_multimodal.py) is exercised with a list of
                #     prompts.
                # Branch on model type rather than guess.
                processor_input = [prompt] if self._model_type == "whisper" else prompt
                inputs = self._processor(processor_input, images=og_images, audios=og_audios)
            except Exception as e:  # pragma: no cover
                del generator
                return self._handle_error("ORT-GenAI multimodal processor failed.", e, "")

            try:
                generator.set_inputs(inputs)
            except RuntimeError as e:
                del generator
                return self._handle_error(
                    "ORT-GenAI generator input setup failed. The prompt may exceed max_length.", e, ""
                )

            # Whisper's BOS == EOS (token 50257 = <|startoftranscript|> = <|endoftext|>),
            # so the very first generated token can collide with EOS. Skip the
            # EOS check until we've emitted at least one non-EOS token.
            decoded = ""
            stream = self._tokenizer.create_stream()
            steps = 0
            generated_any = False
            while not generator.is_done() and steps < max_new_tokens:
                generator.generate_next_token()
                tok = int(generator.get_next_tokens()[0])
                if tok in self._eos_token_ids:
                    if self._model_type != "whisper" or generated_any:
                        break
                    # Whisper's first-step EOS can collide with its BOS token.
                    steps += 1
                    continue
                generated_any = True
                decoded += stream.decode(tok)
                if stop_strings:
                    for s in stop_strings:
                        if s in decoded:
                            decoded = decoded.split(s, 1)[0]
                            del generator
                            return decoded
                steps += 1

        del generator
        return decoded

    def _score_continuation(self, prompt: str, continuation: str, images, audios) -> tuple[float, bool]:
        """Return ``(continuation loss, is_greedy)`` for a loglikelihood request.

        Scores the continuation tokens under the model given ``prompt`` + media.
        lmms-eval's multiple-choice tasks select the minimum returned value, so
        this method returns positive negative-log-likelihood rather than log-probability.
        ``is_greedy`` is True iff every continuation token was the argmax at its step.
        """
        # Tokenize prompt and prompt+continuation jointly, then slice to obtain
        # the continuation token IDs as they would actually appear extending the
        # prompt. Critical for sentencepiece/BPE tokenizers where ``encode("A")``
        # differs from the suffix of ``encode("prompt A")`` (leading-space
        # handling, BOS injection, etc.).
        prompt_tokens = list(self._tokenizer.encode(prompt))
        full_tokens = list(self._tokenizer.encode(prompt + continuation))
        cont_tokens = full_tokens[len(prompt_tokens) :]
        if len(cont_tokens) == 0:
            return 0.0, True

        params = og.GeneratorParams(self._model)
        # `max_length` is total (prompt + completion) including image-embed tokens.
        params.set_search_options(
            max_length=self.max_length,
            do_sample=False,
        )
        generator = og.Generator(self._model, params)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            og_images = self._build_og_images(images, tmp_dir)
            og_audios = self._build_og_audios(audios, tmp_dir)
            try:
                processor_input = [prompt] if self._model_type == "whisper" else prompt
                inputs = self._processor(processor_input, images=og_images, audios=og_audios)
            except Exception as e:  # pragma: no cover
                del generator
                return self._handle_error("ORT-GenAI multimodal processor failed in loglikelihood.", e, (1e9, False))

            try:
                generator.set_inputs(inputs)
            except RuntimeError as e:
                del generator
                return self._handle_error("ORT-GenAI generator input setup failed in loglikelihood.", e, (1e9, False))

            # SetInputs appends the prompt tokens and runs the decoder prefill, so
            # get_logits() immediately returns the distribution for the first
            # continuation token. Appending a chosen continuation token advances
            # the generator and computes logits for the following token.
            total_loss = 0.0
            all_greedy = True
            for i, tok_id in enumerate(cont_tokens):
                if generator.is_done():
                    total_loss += 50.0
                    all_greedy = False
                    continue
                logits = np.asarray(generator.get_logits(), dtype=np.float64).reshape(-1)
                if tok_id >= logits.shape[0]:
                    del generator
                    raise ValueError(f"Token id {tok_id} is outside logits vocabulary size {logits.shape[0]}.")
                log_denom = np.logaddexp.reduce(logits)
                total_loss += float(log_denom - logits[tok_id])
                if int(np.argmax(logits)) != tok_id:
                    all_greedy = False
                if i + 1 < len(cont_tokens):
                    generator.append_tokens(np.array([tok_id], dtype=np.int32))

        del generator
        return total_loss, all_greedy

    _DEFAULT_IMAGE_TOKEN_FORMAT = "<|image_{index}|>"
    _DEFAULT_AUDIO_TOKEN_FORMAT = "<|audio_{index}|>"

    # The generic system prompt used by the majority of lmms-eval wrappers
    # (qwen2_5_vl.py, qwen3_vl.py, gemma3.py, llava_*, minicpm_o, ...). Also the
    # fallback for model types without a more specific mapping.
    _DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    # Qwen Omni models (qwen2_5_omni.py / qwen3_omni.py) are trained with a fixed
    # identity system prompt; reuse it verbatim for prompt parity.
    _QWEN_OMNI_SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    # Per-model-family default system prompts. Each entry is a tuple of substrings
    # that must ALL appear in the (lowercased) genai_config ``model.type``, paired
    # with the prompt to use. First match wins, so more specific rules come first.
    # These mirror lmms-eval's per-model wrappers so HF-input and ORT-GenAI
    # evaluation use identical prompts out of the box. Only applied when the user
    # does not pass an explicit ``system_prompt``.
    #   - Qwen Omni (qwen2_5_omni / qwen3_omni_moe): fixed identity prompt.
    #   - Qwen-VL / Gemma: generic "You are a helpful assistant.".
    #   - Phi-4-MM: lmms-eval's phi4_multimodal wrapper uses NO system turn, so
    #     the parity-correct default is an empty string.
    _MODEL_TYPE_SYSTEM_PROMPTS: tuple[tuple[tuple[str, ...], str], ...] = (
        (("qwen", "omni"), _QWEN_OMNI_SYSTEM_PROMPT),
        (("qwen",), _DEFAULT_SYSTEM_PROMPT),
        (("gemma",), _DEFAULT_SYSTEM_PROMPT),
        (("phi",), ""),
    )

    @classmethod
    def default_system_prompt_for_model_type(cls, model_type: str | None) -> str:
        """Return the parity-correct default system prompt for a model type (see ``_MODEL_TYPE_SYSTEM_PROMPTS``)."""
        model_type_lower = (model_type or "").lower()
        for required_substrings, prompt in cls._MODEL_TYPE_SYSTEM_PROMPTS:
            if all(substring in model_type_lower for substring in required_substrings):
                return prompt
        return cls._DEFAULT_SYSTEM_PROMPT

    def _probe_structured_content_support(self) -> bool:
        """Detect whether the model's chat template correctly injects media tokens from structured content.

        Renders a probe message whose content is a list of typed parts
        (``[{"type": "image"}, {"type": "text", ...}]``) and checks the result:

        - Well-behaved templates (Gemma-4, Qwen2.5-VL, Qwen3-VL) replace the
          image part with the model's own media token (e.g. ``<|image|>`` or
          ``<|vision_start|>...``), so the rendered string contains no Python
          dict repr.
        - Broken templates (Phi-4-MM) stringify the list as Python repr, so the
          rendered string contains ``{'type'`` / ``"type"``.

        Probed once at load. Returns False if the tokenizer has no
        ``apply_chat_template`` or the probe raises.
        """
        if not self._has_chat_template:
            return False
        try:
            probe = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]
            rendered = self._tokenizer.apply_chat_template(json.dumps(probe), add_generation_prompt=True)
            text_only = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
            rendered_text_only = self._tokenizer.apply_chat_template(json.dumps(text_only), add_generation_prompt=True)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("Structured-content probe failed; falling back to pre-render: %s", e)
            return False
        # A useful template must neither leak the content dictionaries nor ignore
        # the image part entirely.
        return "{'type'" not in rendered and '"type"' not in rendered and rendered != rendered_text_only

    def _build_structured_chat_prompt(self, user_text: str, num_images: int, num_audios: int) -> str:
        """Build the prompt via structured content parts so the chat template injects media tokens.

        Only used when :meth:`_probe_structured_content_support` returned True
        and the user did not override the media token formats.
        """
        content: list[dict[str, Any]] = []
        if _MEDIA_PLACEHOLDER_RE.search(user_text):
            image_count = 0
            audio_count = 0
            for part in _MEDIA_PLACEHOLDER_RE.split(user_text):
                if part == "<image>":
                    image_count += 1
                    content.append({"type": "image"})
                elif part == "<audio>":
                    audio_count += 1
                    content.append({"type": "audio"})
                elif part:
                    content.append({"type": "text", "text": part})
            if image_count != num_images or audio_count != num_audios:
                raise ValueError(
                    "Media placeholder counts do not match the extracted media: "
                    f"{image_count} image placeholders for {num_images} images, "
                    f"{audio_count} audio placeholders for {num_audios} audios."
                )
        else:
            content.extend({"type": "image"} for _ in range(num_images))
            content.extend({"type": "audio"} for _ in range(num_audios))
            content.append({"type": "text", "text": user_text})

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return self._tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)

    def _build_prompt_for_request(self, user_text: str, num_images: int, num_audios: int) -> str:
        """Build the final prompt string fed to ``og.MultiModalProcessor``.

        Path selection:

        1. **Whisper**: no chat template; return the decoder-start token sequence.
        2. **Explicit override / no chat template**: legacy ``_build_prompt``
           format-string path (when the user set ``prompt_template`` or the
           onnxruntime-genai version predates ``apply_chat_template``).
        3. **Structured content** (preferred): when the model's chat template
           injects media tokens from structured content parts (auto-detected by
           :meth:`_probe_structured_content_support`) and the user did not set
           ``image_token_format`` / ``audio_token_format``. The template emits
           the correct per-model media tokens (e.g. ``<|image|>`` for Gemma-4,
           ``<|vision_start|>...`` for Qwen2.5-VL) — no per-model config needed.
        4. **Pre-render** (fallback): pre-render media markers into a flat user
           string, then ``apply_chat_template``. Used for templates that
           stringify structured content (Phi-4-MM) or when the user explicitly
           supplies a media token format.
        """
        if self._model_type == "whisper":
            return _build_prompt(
                self._model_type,
                num_images,
                num_audios,
                user_text,
                whisper_language=self.whisper_language,
                whisper_task=self.whisper_task,
                whisper_timestamps=self.whisper_timestamps,
            )

        if self.prompt_template or not self._has_chat_template:
            return _build_prompt(
                self._model_type,
                num_images,
                num_audios,
                user_text,
                self.system_prompt,
                self.prompt_template,
                self.image_token_format or self._DEFAULT_IMAGE_TOKEN_FORMAT,
                self.audio_token_format or self._DEFAULT_AUDIO_TOKEN_FORMAT,
            )

        # Prefer structured content when the template supports it AND the user
        # did not pin a specific media token format. This lets well-behaved
        # templates inject their own correct tokens without per-model config.
        user_pinned_tokens = self.image_token_format is not None or self.audio_token_format is not None
        if self._supports_structured_content and not user_pinned_tokens:
            return self._build_structured_chat_prompt(user_text, num_images, num_audios)

        image_markers = "".join(
            _format_media_tokens(num_images, self.image_token_format or self._DEFAULT_IMAGE_TOKEN_FORMAT)
        )
        audio_markers = "".join(
            _format_media_tokens(num_audios, self.audio_token_format or self._DEFAULT_AUDIO_TOKEN_FORMAT)
        )
        user_content = f"{image_markers}{audio_markers}{user_text}"

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})

        return self._tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)

    def _get_doc_and_visuals(self, doc_to_visual, doc_id, task, split):
        """Look up an lmms-eval document and its visual inputs.

        ``doc_to_visual`` is the task-specific function that extracts images or
        audio from the document.
        """
        try:
            doc = self.task_dict[task][split][doc_id]
        except (KeyError, IndexError, TypeError) as e:
            raise KeyError(f"Failed to find lmms-eval document={task!r}, split={split!r}, doc_id={doc_id!r}") from e

        visuals = doc_to_visual(doc) if doc_to_visual else []
        if visuals is None:
            visuals = []
        if not isinstance(visuals, list):
            visuals = [visuals]
        return doc, visuals

    # -------------------------------------------------------------------------
    # lmms-eval Model interface
    # -------------------------------------------------------------------------
    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        """lmms-eval entry point: generate a text completion for each request.

        Per request, resolves media, the ``max_new_tokens`` budget (task YAML wins),
        and ``until`` stop strings (minus ``ignore_stop_strings``), builds the prompt,
        and greedily decodes. Returns one string per request, in order.
        """
        results = []
        pbar = tqdm(total=len(requests), desc="ortgenai_mm generate_until", disable=disable_tqdm)
        for req in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            # NOTE, visuals could contain both images and audio
            _, visuals = self._get_doc_and_visuals(doc_to_visual, doc_id, task, split)
            images, audios = _partition_visuals(visuals)

            gen_kwargs = gen_kwargs or {}
            max_new_tokens = int(gen_kwargs.get("max_new_tokens", self.max_new_tokens))
            search_options, stop = _generation_options(gen_kwargs)
            if stop and self.ignore_stop_strings:
                stop = [s for s in stop if s not in self.ignore_stop_strings] or None

            prompt = self._build_prompt_for_request(contexts, len(images), len(audios))
            text = self._run_generation(
                prompt,
                images,
                audios,
                max_new_tokens,
                stop,
                search_options=search_options,
            )
            results.append(text)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), text)
            pbar.update(1)
        pbar.close()
        return results

    def loglikelihood(self, requests: list[Instance], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        """lmms-eval entry point: score each request's target continuation.

        Returns one ``(loss, is_greedy)`` per request. lmms-eval multiple-choice
        tasks select the candidate with the minimum loss.
        """
        results = []
        pbar = tqdm(total=len(requests), desc="ortgenai_mm loglikelihood", disable=disable_tqdm)
        for req in requests:
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = req.args
            doc, visuals = self._get_doc_and_visuals(doc_to_visual, doc_id, task, split)
            images, audios = _partition_visuals(visuals)
            continuation = doc_to_target if isinstance(doc_to_target, str) else str(doc_to_target(doc))

            prompt = self._build_prompt_for_request(contexts, len(images), len(audios))
            loss, is_greedy = self._score_continuation(prompt, continuation, images, audios)
            results.append((loss, is_greedy))
            self.cache_hook.add_partial("loglikelihood", (contexts, continuation), (loss, is_greedy))
            pbar.update(1)
        pbar.close()
        return results

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("ortgenai_mm does not support lmms-eval multi-round generation yet.")
