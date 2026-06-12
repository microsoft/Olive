# -------------------------------------------------------------------------
# lmms-eval adapter for Olive-exported multimodal models.
#
# Added locally (not upstream) to support evaluating quantized multimodal ONNX
# models through the EvolvingLMMs-Lab/lmms-eval harness, mirroring how
# olive/evaluator/lmeval_ort.py wraps lm-evaluation-harness for text models.
#
# Registers an LMMSORTGenAIEvaluator class with lmms-eval's legacy
# @register_model registry under the name "ortgenai_mm". Consumers obtain it
# via lmms_eval.api.registry.get_model("ortgenai_mm").
# -------------------------------------------------------------------------
"""lmms-eval ORT-GenAI adapter for Olive-exported multimodal models."""

from __future__ import annotations

import io
import json
import logging
import tempfile
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


def _normalize_image(visual) -> PIL.Image.Image | None:
    if isinstance(visual, PIL.Image.Image):
        return visual.convert("RGB")
    if isinstance(visual, (str, Path)):
        p = Path(visual)
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            return PIL.Image.open(p).convert("RGB")
        return None
    if isinstance(visual, dict):
        if "bytes" in visual:
            return PIL.Image.open(io.BytesIO(visual["bytes"])).convert("RGB")
        if "path" in visual:
            return PIL.Image.open(visual["path"]).convert("RGB")
    if isinstance(visual, np.ndarray):
        return PIL.Image.fromarray(np.uint8(visual)).convert("RGB")
    return None


def _normalize_audio(visual) -> tuple[np.ndarray, int] | None:
    if isinstance(visual, dict) and "array" in visual and "sampling_rate" in visual:
        return np.asarray(visual["array"], dtype=np.float32), int(visual["sampling_rate"])
    if isinstance(visual, (str, Path)):
        p = Path(visual)
        if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            try:
                import librosa
            except ImportError:
                logger.warning("Audio file %s encountered but librosa not installed.", p)
                return None
            arr, sr = librosa.load(str(p), sr=None, mono=True)
            return arr.astype(np.float32), int(sr)
    return None


def _partition_visuals(visuals):
    images, audios = [], []
    for v in visuals or []:
        if v is None:
            continue
        img = _normalize_image(v)
        if img is not None:
            images.append(img)
            continue
        au = _normalize_audio(v)
        if au is not None:
            audios.append(au)
    return images, audios


def _format_media_tokens(num_items: int, token_format: str) -> list[str]:
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
) -> str:
    """Build a Phi-4-multimodal-style chat prompt.

    Other multimodal architectures use different placeholder tags. Users can
    override the media token formats and the full prompt template from the Olive
    evaluator config without changing this adapter.
    """
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
    r"""lmms-eval model wrapper for an ORT-GenAI multimodal package.

    Example::

        lmms_eval --model ortgenai_mm \\
            --model_args pretrained=/path/to/ort_genai_dir,batch_size=1 \\
            --tasks mmmu_val --limit 4
    """

    is_simple = True

    def __init__(
        self,
        pretrained: str,
        batch_size: int = 1,
        max_new_tokens: int = 256,
        max_length: int = 8192,
        system_prompt: str = "You are a helpful AI assistant.",
        execution_provider: str | None = None,
        provider_options: dict | None = None,
        fail_on_error: bool = True,
        prompt_template: str | None = None,
        image_token_format: str = "<|image_{index}|>",
        audio_token_format: str = "<|audio_{index}|>",
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

        eos_ids = self._tokenizer.eos_token_ids
        self._eos_token_ids = {int(t) for t in (eos_ids if eos_ids is not None else [])}

        try:
            cfg = json.loads((Path(self.model_dir) / "genai_config.json").read_text(encoding="utf-8"))
            self._model_type = cfg.get("model", {}).get("type", "phi4mm")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid genai_config.json in {self.model_dir}") from e

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
        if not images:
            return None
        paths = []
        for i, img in enumerate(images):
            path = tmp_dir / f"image_{i}.png"
            img.save(path, format="PNG")
            paths.append(str(path))
        return og.Images.open(*paths)

    def _build_og_audios(self, audios, tmp_dir: Path):
        if not audios:
            return None
        import soundfile as sf

        paths = []
        for i, (arr, sr) in enumerate(audios):
            path = tmp_dir / f"audio_{i}.wav"
            sf.write(path, arr, sr)
            paths.append(str(path))
        return og.Audios.open(*paths)

    def _handle_error(self, message: str, exc: Exception, default):
        if self.fail_on_error:
            raise RuntimeError(message) from exc
        logger.exception("%s", message)
        return default

    # -------------------------------------------------------------------------
    # Single-request inference primitives
    # -------------------------------------------------------------------------
    def _run_generation(
        self, prompt: str, images, audios, max_new_tokens: int, stop_strings: list[str] | None = None
    ) -> str:
        params = og.GeneratorParams(self._model)
        # `max_length` is total (prompt + completion). Image prompts can be huge
        # (Phi-4-MM image embeds are 1000+ tokens), so default generously.
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
                inputs = self._processor(prompt, images=og_images, audios=og_audios)
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

            decoded = ""
            stream = self._tokenizer.create_stream()
            steps = 0
            while not generator.is_done() and steps < max_new_tokens:
                generator.generate_next_token()
                tok = int(generator.get_next_tokens()[0])
                if tok in self._eos_token_ids:
                    break
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
                inputs = self._processor(prompt, images=og_images, audios=og_audios)
            except Exception as e:  # pragma: no cover
                del generator
                return self._handle_error("ORT-GenAI multimodal processor failed in loglikelihood.", e, (-1e9, False))

            try:
                generator.set_inputs(inputs)
            except RuntimeError as e:
                del generator
                return self._handle_error("ORT-GenAI generator input setup failed in loglikelihood.", e, (-1e9, False))

            # ORT-GenAI's ``set_inputs`` only loads the prompt + multimodal embeds;
            # it does NOT run the decoder forward pass. ``get_logits()`` therefore
            # returns an undefined buffer before any compute step. Trigger the
            # prompt-fill forward pass with ``generate_next_token()`` (the sampled
            # token is discarded via ``rewind_to`` after the first scoring
            # iteration, before our chosen continuation token is appended).
            token_count_after_prefill = generator.token_count()
            generator.generate_next_token()

            total_logprob = 0.0
            all_greedy = True
            for i, tok_id in enumerate(cont_tokens):
                if generator.is_done():
                    total_logprob += -50.0
                    all_greedy = False
                    continue
                logits = np.asarray(generator.get_logits(), dtype=np.float64).reshape(-1)
                if tok_id >= logits.shape[0]:
                    del generator
                    raise ValueError(f"Token id {tok_id} is outside logits vocabulary size {logits.shape[0]}.")
                log_denom = np.logaddexp.reduce(logits)
                total_logprob += float(logits[tok_id] - log_denom)
                if int(np.argmax(logits)) != tok_id:
                    all_greedy = False
                if i == 0:
                    # Drop the throwaway token sampled by ``generate_next_token``
                    # above so ``append_tokens`` lands at end-of-prompt + cont[0],
                    # not end-of-prompt + sampled + cont[0].
                    generator.rewind_to(token_count_after_prefill)
                generator.append_tokens(np.array([tok_id], dtype=np.int32))

        del generator
        return total_logprob, all_greedy

    def _get_doc_and_visuals(self, doc_to_visual, doc_id, task, split):
        try:
            doc = self.task_dict[task][split][doc_id]
        except (KeyError, IndexError, TypeError) as e:
            raise KeyError(
                f"Failed to find lmms-eval document task={task!r}, split={split!r}, doc_id={doc_id!r}"
            ) from e

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
        results = []
        pbar = tqdm(total=len(requests), desc="ortgenai_mm generate_until", disable=disable_tqdm)
        for req in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            _, visuals = self._get_doc_and_visuals(doc_to_visual, doc_id, task, split)
            images, audios = _partition_visuals(visuals)

            gen_kwargs = gen_kwargs or {}
            max_new = int(gen_kwargs.get("max_new_tokens", self.max_new_tokens))
            stop = gen_kwargs.get("until", None)
            if isinstance(stop, str):
                stop = [stop]

            prompt = _build_prompt(
                self._model_type,
                len(images),
                len(audios),
                contexts,
                self.system_prompt,
                self.prompt_template,
                self.image_token_format,
                self.audio_token_format,
            )
            text = self._run_generation(prompt, images, audios, max_new, stop)
            results.append(text)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), text)
            pbar.update(1)
        pbar.close()
        return results

    def loglikelihood(self, requests: list[Instance], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        results = []
        pbar = tqdm(total=len(requests), desc="ortgenai_mm loglikelihood", disable=disable_tqdm)
        for req in requests:
            contexts, doc_to_target, doc_to_visual, doc_id, task, split = req.args
            doc, visuals = self._get_doc_and_visuals(doc_to_visual, doc_id, task, split)
            images, audios = _partition_visuals(visuals)
            continuation = str(doc_to_target(doc))

            prompt = _build_prompt(
                self._model_type,
                len(images),
                len(audios),
                contexts,
                self.system_prompt,
                self.prompt_template,
                self.image_token_format,
                self.audio_token_format,
            )
            logprob, is_greedy = self._score_continuation(prompt, continuation, images, audios)
            results.append((logprob, is_greedy))
            self.cache_hook.add_partial("loglikelihood", (contexts, continuation), (logprob, is_greedy))
            pbar.update(1)
        pbar.close()
        return results

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("ortgenai_mm does not support lmms-eval multi-round generation yet.")


# -----------------------------------------------------------------------------
# lmms-eval MODEL_REGISTRY_V2 entry-point factory.
#
# Exposed via setup.py entry_points["lmms_eval.models"], so a fresh install of
# olive-ai makes ``--model ortgenai_mm`` discoverable from the lmms-eval CLI
# (e.g. ``python -m lmms_eval --model ortgenai_mm ...``) without requiring the
# caller to import this module first.
#
# lmms-eval's ``ModelRegistryV2.load_entrypoint_manifests`` accepts a
# ``Callable`` payload, so we keep the import of ``ModelManifest`` lazy. That
# way ``olive`` (and the rest of this module) stays importable when lmms-eval
# is not installed.
# -----------------------------------------------------------------------------
def _model_manifest():
    """Return the lmms-eval ModelManifest for ``ortgenai_mm``.

    Used as an entry-point payload for lmms-eval's MODEL_REGISTRY_V2.
    """
    from lmms_eval.models.registry_v2 import ModelManifest

    return ModelManifest(
        model_id="ortgenai_mm",
        simple_class_path="olive.evaluator.lmms_ort.LMMSORTGenAIEvaluator",
    )
