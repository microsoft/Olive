# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Tests intentionally exercise "protected" runtime methods (_score_continuation,
# _run_generation) and configure fake collaborators by setting attributes
# directly on the fake. Both are normal in unit tests, so suppress pylint's
# protected-access / attribute-defined-outside-init warnings for this file.
# pylint: disable=protected-access,attribute-defined-outside-init
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest

from olive.evaluator.lmms_ort import (
    LMMSORTGenAIEvaluator,
    _build_prompt,
    _normalize_execution_provider,
)
from olive.evaluator.olive_evaluator import LMMSEvaluator
from olive.model import ONNXModelHandler


def test_build_prompt_uses_default_phi4mm_tokens():
    prompt = _build_prompt("phi4mm", 1, 1, "What happened?", "System prompt.")

    assert prompt == "<|system|>System prompt.<|end|><|user|><|image_1|><|audio_1|>What happened?<|end|><|assistant|>"


def test_build_prompt_uses_custom_template_and_token_formats():
    prompt = _build_prompt(
        "custom",
        2,
        1,
        "Question",
        "System",
        prompt_template="{system_prompt}\n{image_tokens}{audio_tokens}\n{text}",
        image_token_format="<image:{zero_index}>",
        audio_token_format="<audio:{index}>",
    )

    assert prompt == "System\n<image:0><image:1><audio:1>\nQuestion"


@pytest.mark.parametrize(
    ("execution_provider", "expected"),
    [
        ("CUDAExecutionProvider", "cuda"),
        ("CPUExecutionProvider", "cpu"),
        ("DmlExecutionProvider", "dml"),
        ("gpu", "cuda"),
        (None, "follow_config"),
        (("CUDAExecutionProvider", {"device_id": "0"}), "cuda"),
    ],
)
def test_normalize_execution_provider(execution_provider, expected):
    assert _normalize_execution_provider(execution_provider) == expected


def _make_evaluator_for_prompt_tests():
    """Construct an LMMSORTGenAIEvaluator with __init__ skipped.

    Lets us unit-test the prompt-building path without needing a real ORT-GenAI
    model on disk.
    """
    inst = LMMSORTGenAIEvaluator.__new__(LMMSORTGenAIEvaluator)
    inst._tokenizer = MagicMock(name="og.Tokenizer")
    inst._model_type = "test_model"
    inst.system_prompt = "You are helpful."
    inst.prompt_template = None
    inst.image_token_format = "<|image_{index}|>"
    inst.audio_token_format = "<|audio_{index}|>"
    inst._has_chat_template = True
    return inst


def test_build_prompt_for_request_uses_apply_chat_template_by_default():
    inst = _make_evaluator_for_prompt_tests()
    inst._tokenizer.apply_chat_template.return_value = "<rendered chat prompt>"

    out = inst._build_prompt_for_request("What is in the image?", num_images=1, num_audios=0)

    assert out == "<rendered chat prompt>"
    inst._tokenizer.apply_chat_template.assert_called_once()
    messages_json_arg = inst._tokenizer.apply_chat_template.call_args.args[0]
    assert inst._tokenizer.apply_chat_template.call_args.kwargs.get("add_generation_prompt") is True

    import json as _json

    messages = _json.loads(messages_json_arg)
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "<|image_1|>What is in the image?"}


def test_build_prompt_for_request_skips_system_when_empty():
    inst = _make_evaluator_for_prompt_tests()
    inst.system_prompt = ""
    inst._tokenizer.apply_chat_template.return_value = "out"

    inst._build_prompt_for_request("Q", num_images=0, num_audios=0)

    import json as _json

    messages = _json.loads(inst._tokenizer.apply_chat_template.call_args.args[0])
    assert all(m["role"] != "system" for m in messages)
    assert messages[-1] == {"role": "user", "content": "Q"}


def test_build_prompt_for_request_includes_audio_markers():
    inst = _make_evaluator_for_prompt_tests()
    inst.system_prompt = ""
    inst._tokenizer.apply_chat_template.return_value = "out"

    inst._build_prompt_for_request("Q", num_images=2, num_audios=1)

    import json as _json

    messages = _json.loads(inst._tokenizer.apply_chat_template.call_args.args[0])
    assert messages[0]["content"] == "<|image_1|><|image_2|><|audio_1|>Q"


def test_build_prompt_for_request_falls_back_to_legacy_when_prompt_template_set():
    inst = _make_evaluator_for_prompt_tests()
    inst.prompt_template = "{system_prompt}|{user_content}"

    out = inst._build_prompt_for_request("Q", num_images=1, num_audios=0)

    assert out == "You are helpful.|<|image_1|>Q"
    inst._tokenizer.apply_chat_template.assert_not_called()


def test_build_prompt_for_request_falls_back_when_chat_template_unavailable():
    inst = _make_evaluator_for_prompt_tests()
    inst._has_chat_template = False  # simulate older onnxruntime-genai

    out = inst._build_prompt_for_request("Q", num_images=1, num_audios=0)

    # Default legacy template wraps with Phi-4-MM-style tokens.
    assert "<|image_1|>" in out
    assert "Q" in out
    inst._tokenizer.apply_chat_template.assert_not_called()


def test_lmms_evaluator_converts_lmms_results(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "text.onnx"
    model_path.touch()
    (model_dir / "genai_config.json").write_text('{"model": {"type": "phi4mm"}}', encoding="utf-8")

    output_path = tmp_path / "results.json"
    evaluator = LMMSEvaluator(
        tasks=["ai2d_lite"],
        batch_size=1,
        limit=2,
        output_path=str(output_path),
        fail_on_error=False,
        prompt_template="{user_content}",
        image_token_format="<image>",
    )
    model = ONNXModelHandler(model_path=str(model_path))

    simple_evaluate_result = {
        "results": {
            "ai2d_lite": {
                "alias": "AI2D Lite",
                "exact_match,none": 0.5,
                "exact_match_stderr,none": 0.1,
                "samples": [{"ignored": True}],
            }
        },
        "configs": {"ai2d_lite": {"task": "ai2d_lite"}},
    }

    simple_evaluate_mock = MagicMock(return_value=simple_evaluate_result)
    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = simple_evaluate_mock

    with (
        patch.dict(
            sys.modules,
            {"lmms_eval": lmms_eval_module, "lmms_eval.evaluator": lmms_eval_evaluator_module},
        ),
        patch("olive.evaluator.lmms_ort.LMMSORTGenAIEvaluator", return_value=SimpleNamespace()) as lm_mock,
    ):
        result = evaluator.evaluate(model, [], execution_providers=["CUDAExecutionProvider"])

    lm_mock.assert_called_once_with(
        pretrained=str(model_dir),
        batch_size=1,
        max_new_tokens=256,
        max_length=32768,
        system_prompt="You are a helpful AI assistant.",
        execution_provider="CUDAExecutionProvider",
        provider_options=None,
        fail_on_error=False,
        prompt_template="{user_content}",
        image_token_format="<image>",
        audio_token_format="<|audio_{index}|>",
    )
    simple_evaluate_mock.assert_called_once()
    assert result.get_value("ai2d_lite", "exact_match") == 0.5
    assert output_path.exists()


def test_lmms_evaluator_requires_genai_config(tmp_path):
    model_path = tmp_path / "text.onnx"
    model_path.touch()
    evaluator = LMMSEvaluator(tasks=["ai2d_lite"])
    model = ONNXModelHandler(model_path=str(model_path))

    with pytest.raises(ValueError, match="requires an ORT-GenAI package"):
        evaluator.evaluate(model, [])


# -----------------------------------------------------------------------------
# HuggingFace dispatch
# -----------------------------------------------------------------------------


def _make_hf_model_handler_stub(model_name_or_path: str, hf_model_type: str):
    handler = MagicMock(name="HfModelHandler")
    handler.model_name_or_path = model_name_or_path
    handler.get_hf_model_type.return_value = hf_model_type
    return handler


def _patch_isinstance_for_hf(handler_stub, monkeypatch):
    """Force isinstance(handler_stub, HfModelHandler) to True for the test path.

    Avoids constructing a real HfModelHandler (which would require a real HF
    model on disk) while still exercising the dispatch logic.
    """
    import olive.evaluator.olive_evaluator as oe

    real_isinstance = isinstance

    def _isinstance(obj, cls):
        if obj is handler_stub and cls is oe.HfModelHandler:
            return True
        if obj is handler_stub and cls is oe.ONNXModelHandler:
            return False
        return real_isinstance(obj, cls)

    monkeypatch.setattr(oe, "isinstance", _isinstance, raising=False)
    oe.isinstance = _isinstance


class _FakePhi4Wrapper:
    """Fake lmms-eval wrapper with a phi4_multimodal-style signature.

    Accepts dtype + trust_remote_code (mirrors lmms-eval's phi4_multimodal class).
    """

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, pretrained, device="cuda", dtype="auto", batch_size=1, trust_remote_code=True, **kwargs):
        type(self).last_kwargs = {
            "pretrained": pretrained,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }


class _FakeQwenWrapper:
    """Fake lmms-eval wrapper with a qwen2_5_vl-style signature.

    Does NOT accept dtype or trust_remote_code (mirrors lmms-eval's qwen2_5_vl
    class which asserts kwargs == {}).
    """

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, pretrained, device="cuda", device_map="auto", batch_size=1, **kwargs):
        if kwargs:
            raise AssertionError(f"Unexpected kwargs: {kwargs}")
        type(self).last_kwargs = {
            "pretrained": pretrained,
            "device": device,
            "device_map": device_map,
            "batch_size": batch_size,
        }


class _FakeKwargsWrapper:
    """Fake lmms-eval wrapper that absorbs ALL options via ``**kwargs``.

    Mirrors lmms-eval wrappers (and HF model wrappers more generally) that take
    only the required ``pretrained`` argument by name and pass everything else
    through ``**kwargs`` to the underlying HF transformers model. Used to verify
    LMMSEvaluator forwards optional kwargs (dtype, trust_remote_code, device) to
    such wrappers instead of silently dropping them because they aren't in
    ``inspect.signature(...).parameters`` as named params.
    """

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, pretrained, **kwargs):
        type(self).last_kwargs = {"pretrained": pretrained, **kwargs}


def test_lmms_evaluator_auto_detects_hf_model_class_from_model_type(tmp_path, monkeypatch):
    """When model_class is unset, auto-detect from HfModelHandler.get_hf_model_type()."""
    handler_stub = _make_hf_model_handler_stub("/local/path/Phi-4-multimodal-instruct", "phi4mm")
    _patch_isinstance_for_hf(handler_stub, monkeypatch)

    output_path = tmp_path / "results.json"
    evaluator = LMMSEvaluator(tasks=["ai2d_lite"], batch_size=2, limit=4, output_path=str(output_path))

    simple_evaluate_mock = MagicMock(
        return_value={"results": {"ai2d_lite": {"alias": "AI2D", "exact_match,none": 0.75}}, "configs": {}}
    )
    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = simple_evaluate_mock
    lmms_eval_models_module = ModuleType("lmms_eval.models")
    lmms_eval_models_module.get_model = MagicMock(return_value=_FakePhi4Wrapper)

    with patch.dict(
        sys.modules,
        {
            "lmms_eval": lmms_eval_module,
            "lmms_eval.evaluator": lmms_eval_evaluator_module,
            "lmms_eval.models": lmms_eval_models_module,
        },
    ):
        result = evaluator.evaluate(handler_stub, [])

    lmms_eval_models_module.get_model.assert_called_once_with("phi4_multimodal")
    assert _FakePhi4Wrapper.last_kwargs["pretrained"] == "/local/path/Phi-4-multimodal-instruct"
    assert _FakePhi4Wrapper.last_kwargs["batch_size"] == 2
    # trust_remote_code defaults to False (see olive/evaluator/olive_evaluator.py
    # LMMSEvaluator.__init__); users opt in explicitly in the recipe.
    assert _FakePhi4Wrapper.last_kwargs["trust_remote_code"] is False
    assert _FakePhi4Wrapper.last_kwargs["dtype"] == "auto"
    simple_evaluate_mock.assert_called_once()
    assert result.get_value("ai2d_lite", "exact_match") == 0.75


def test_lmms_evaluator_filters_kwargs_for_qwen_style_wrapper(monkeypatch):
    """Wrappers like qwen2_5_vl reject unknown kwargs.

    LMMSEvaluator must inspect the wrapper signature and only forward kwargs
    the wrapper actually declares as named parameters.
    """
    handler_stub = _make_hf_model_handler_stub("/p/Qwen2.5-VL-3B-Instruct", "qwen2_5_vl")
    _patch_isinstance_for_hf(handler_stub, monkeypatch)

    evaluator = LMMSEvaluator(tasks=["mmstar"], batch_size=1, dtype="bfloat16", trust_remote_code=True)

    simple_evaluate_mock = MagicMock(return_value={"results": {}, "configs": {}})
    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = simple_evaluate_mock
    lmms_eval_models_module = ModuleType("lmms_eval.models")
    lmms_eval_models_module.get_model = MagicMock(return_value=_FakeQwenWrapper)

    with patch.dict(
        sys.modules,
        {
            "lmms_eval": lmms_eval_module,
            "lmms_eval.evaluator": lmms_eval_evaluator_module,
            "lmms_eval.models": lmms_eval_models_module,
        },
    ):
        evaluator.evaluate(handler_stub, [])

    # dtype + trust_remote_code must NOT have been forwarded (Qwen wrapper would error)
    assert "dtype" not in _FakeQwenWrapper.last_kwargs
    assert "trust_remote_code" not in _FakeQwenWrapper.last_kwargs
    # but pretrained, device, batch_size MUST have been forwarded
    assert _FakeQwenWrapper.last_kwargs["pretrained"] == "/p/Qwen2.5-VL-3B-Instruct"
    assert _FakeQwenWrapper.last_kwargs["device"] == "cpu"  # Device.CPU default
    assert _FakeQwenWrapper.last_kwargs["batch_size"] == 1


def test_lmms_evaluator_does_not_forward_to_pure_var_keyword_wrappers(monkeypatch):
    """Verify ``device``/``dtype``/``trust_remote_code`` are NOT auto-forwarded to ``**kwargs`` wrappers.

    Rationale: the signature alone cannot tell "absorbs unknowns" from "rejects
    unknowns at runtime" (qwen2_5_vl has ``**kwargs`` and asserts ``kwargs ==
    {}``). To stay safe, only kwargs named explicitly as parameters are
    auto-forwarded. Users who want to forward additional kwargs to a pure
    ``**kwargs`` wrapper must use ``hf_model_kwargs`` (explicit user opt-in).
    """
    handler_stub = _make_hf_model_handler_stub("/p/some-vlm", "qwen2_5_vl")
    _patch_isinstance_for_hf(handler_stub, monkeypatch)

    evaluator = LMMSEvaluator(
        tasks=["mmstar"],
        batch_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        # The escape hatch for forwarding arbitrary kwargs to a wrapper that
        # absorbs them via **kwargs:
        hf_model_kwargs={"custom_backend_opt": "value"},
    )

    simple_evaluate_mock = MagicMock(return_value={"results": {}, "configs": {}})
    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = simple_evaluate_mock
    lmms_eval_models_module = ModuleType("lmms_eval.models")
    lmms_eval_models_module.get_model = MagicMock(return_value=_FakeKwargsWrapper)

    with patch.dict(
        sys.modules,
        {
            "lmms_eval": lmms_eval_module,
            "lmms_eval.evaluator": lmms_eval_evaluator_module,
            "lmms_eval.models": lmms_eval_models_module,
        },
    ):
        evaluator.evaluate(handler_stub, [])

    # Required kwargs are always forwarded.
    assert _FakeKwargsWrapper.last_kwargs["pretrained"] == "/p/some-vlm"
    assert _FakeKwargsWrapper.last_kwargs["batch_size"] == 1
    # Optional kwargs are NOT forwarded to pure **kwargs wrappers.
    assert "dtype" not in _FakeKwargsWrapper.last_kwargs
    assert "trust_remote_code" not in _FakeKwargsWrapper.last_kwargs
    assert "device" not in _FakeKwargsWrapper.last_kwargs
    # The explicit hf_model_kwargs escape hatch IS forwarded.
    assert _FakeKwargsWrapper.last_kwargs["custom_backend_opt"] == "value"


def test_lmms_evaluator_uses_explicit_model_class_when_set(monkeypatch):
    """An explicit ``model_class`` in the recipe overrides auto-detection."""
    handler_stub = _make_hf_model_handler_stub("/p/some-vlm", "some-unknown-vlm-type")
    _patch_isinstance_for_hf(handler_stub, monkeypatch)

    evaluator = LMMSEvaluator(tasks=["mmstar"], model_class="qwen2_5_vl", batch_size=1)

    simple_evaluate_mock = MagicMock(return_value={"results": {}, "configs": {}})
    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = simple_evaluate_mock
    lmms_eval_models_module = ModuleType("lmms_eval.models")
    lmms_eval_models_module.get_model = MagicMock(return_value=_FakeQwenWrapper)

    with patch.dict(
        sys.modules,
        {
            "lmms_eval": lmms_eval_module,
            "lmms_eval.evaluator": lmms_eval_evaluator_module,
            "lmms_eval.models": lmms_eval_models_module,
        },
    ):
        evaluator.evaluate(handler_stub, [])

    lmms_eval_models_module.get_model.assert_called_once_with("qwen2_5_vl")
    handler_stub.get_hf_model_type.assert_not_called()


def test_lmms_evaluator_raises_when_hf_model_type_is_unmapped(monkeypatch):
    """If we can't auto-detect and the user didn't set model_class, fail loudly."""
    handler_stub = _make_hf_model_handler_stub("/p/exotic-model", "some-exotic-vlm")
    _patch_isinstance_for_hf(handler_stub, monkeypatch)

    evaluator = LMMSEvaluator(tasks=["mmstar"])

    # Even with lmms_eval modules mocked, the error fires before reaching them.
    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = MagicMock()
    lmms_eval_models_module = ModuleType("lmms_eval.models")
    lmms_eval_models_module.get_model = MagicMock()
    with (
        patch.dict(
            sys.modules,
            {
                "lmms_eval": lmms_eval_module,
                "lmms_eval.evaluator": lmms_eval_evaluator_module,
                "lmms_eval.models": lmms_eval_models_module,
            },
        ),
        pytest.raises(ValueError, match=r"Could not auto-detect lmms-eval model_class"),
    ):
        evaluator.evaluate(handler_stub, [])


def test_lmms_evaluator_rejects_unsupported_handler_type():
    """LMMSEvaluator only supports HfModelHandler and ONNXModelHandler-as-ortgenai."""
    evaluator = LMMSEvaluator(tasks=["mmstar"])
    bogus = SimpleNamespace()  # neither HfModelHandler nor ONNXModelHandler

    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = MagicMock()
    with (
        patch.dict(sys.modules, {"lmms_eval": lmms_eval_module, "lmms_eval.evaluator": lmms_eval_evaluator_module}),
        pytest.raises(ValueError, match=r"ONNXModelHandler.*HfModelHandler"),
    ):
        evaluator.evaluate(bogus, [])


# -----------------------------------------------------------------------------
# lmms-eval MODEL_REGISTRY_V2 entry-point integration
# -----------------------------------------------------------------------------


def test_lmms_ort_genai_evaluator_is_simple_flag_matches_registration():
    """Verify is_simple matches the lmms-eval registration type.

    MODEL_REGISTRY_V2._validate_model_class requires the class' ``is_simple``
    flag to match the registered model_type (``simple`` vs ``chat``). Our
    adapter is registered with ``simple_class_path``, so ``is_simple`` must
    be ``True``.
    """
    assert LMMSORTGenAIEvaluator.is_simple is True


# -----------------------------------------------------------------------------
# Visual partitioning
# -----------------------------------------------------------------------------


def test_partition_visuals_separates_images_and_audios_and_skips_nones():
    from olive.evaluator.lmms_ort import _partition_visuals

    img1 = PIL.Image.new("RGB", (4, 4))
    img2_dict = {"bytes": _png_bytes(PIL.Image.new("RGB", (2, 2)))}
    audio_dict = {"array": np.zeros(16, dtype=np.float32), "sampling_rate": 16000}

    images, audios = _partition_visuals([img1, None, img2_dict, audio_dict, None])

    assert len(images) == 2
    assert all(isinstance(img, PIL.Image.Image) for img in images)
    assert len(audios) == 1
    arr, sr = audios[0]
    assert arr.shape == (16,)
    assert sr == 16000


def test_partition_visuals_handles_none_input():
    from olive.evaluator.lmms_ort import _partition_visuals

    assert _partition_visuals(None) == ([], [])
    assert _partition_visuals([]) == ([], [])


def _png_bytes(image):
    import io as _io

    buf = _io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Runtime: _score_continuation and _run_generation
#
# These tests construct a LMMSORTGenAIEvaluator with the onnxruntime_genai
# module wholesale-mocked, so the generation/scoring flows are exercised
# end-to-end without needing an actual ORT-GenAI model on disk.
# -----------------------------------------------------------------------------


class _FakeTokenStream:
    def decode(self, tok):
        return f"<t{tok}>"


class _FakeTokenizer:
    """Minimal stub for og.Tokenizer used by _score_continuation/_run_generation.

    ``tokens_for`` is a dict mapping the exact input string to the list of
    token ids encode() should return; this lets each test inject specific
    prompt + (prompt+continuation) tokenizations to drive the slicing logic.
    """

    def __init__(self, tokens_for, eos_token_ids=(99,)):
        self._tokens_for = tokens_for
        self.eos_token_ids = list(eos_token_ids)

    def encode(self, text):
        if text not in self._tokens_for:
            raise KeyError(f"_FakeTokenizer: no canned tokenization for text={text!r}")
        return list(self._tokens_for[text])

    def create_stream(self):
        return _FakeTokenStream()


class _FakeGenerator:
    """Records call order so tests can assert the score/generation protocol.

    ``logits_queue`` is a list of numpy arrays consumed in order, one per forward
    pass: ``generate_next_token`` and ``append_tokens`` each consume one entry
    into ``_current_logits``, which ``get_logits()`` returns.
    """

    instances: ClassVar[list] = []

    def __init__(self, model, params):
        type(self).instances.append(self)
        self._model = model
        self._params = params
        self._logits_queue = list(model._next_logits_queue)
        self._sampled_queue = list(model._next_sampled_queue)
        self._current_logits = None
        self._token_count = 0
        self._done = False
        self._last_sampled = -1
        self.calls = []

    def _consume_forward_pass(self):
        if not self._logits_queue:
            raise RuntimeError("_FakeGenerator: forward pass exhausted (logits_queue empty)")
        self._current_logits = self._logits_queue.pop(0)

    def set_inputs(self, inputs):
        # set_inputs only loads inputs; it does NOT trigger a forward pass and
        # therefore does NOT populate _current_logits. This is the behavior the
        # production code at lmms_ort.py:_score_continuation has to compensate
        # for by calling generate_next_token() to force the prompt-fill compute.
        self.calls.append(("set_inputs", inputs))

    def generate_next_token(self):
        self._consume_forward_pass()
        sampled = self._sampled_queue.pop(0) if self._sampled_queue else -1
        self._last_sampled = sampled
        self._token_count += 1
        self.calls.append(("generate_next_token", sampled))

    def get_logits(self):
        if self._current_logits is None:
            raise RuntimeError("_FakeGenerator: get_logits called before any forward pass")
        return np.asarray(self._current_logits, dtype=np.float32)

    def get_next_tokens(self):
        return np.array([self._last_sampled], dtype=np.int32)

    def append_tokens(self, tok_array):
        toks = [int(t) for t in np.asarray(tok_array).reshape(-1)]
        self.calls.append(("append_tokens", toks))
        self._token_count += len(toks)
        # Each appended token batch is one forward pass (computes new last-position logits).
        self._consume_forward_pass()

    def token_count(self):
        return self._token_count

    def is_done(self):
        return self._done

    def rewind_to(self, n):
        self.calls.append(("rewind_to", n))
        self._token_count = n


class _FakeGeneratorParams:
    def __init__(self, model):
        self._model = model

    def set_search_options(self, **kwargs):
        pass


class _FakeProcessor:
    def __call__(self, prompt, images=None, audios=None):
        return SimpleNamespace(_prompt=prompt, _images=images, _audios=audios)


class _FakeOgModel:
    def __init__(self, tokenizer=None, next_logits_queue=None, next_sampled_queue=None):
        self.tokenizer = tokenizer
        self._next_logits_queue = list(next_logits_queue or [])
        self._next_sampled_queue = list(next_sampled_queue or [])

    def create_multimodal_processor(self):
        return _FakeProcessor()


def _make_fake_og(model):
    """Build a SimpleNamespace mimicking the onnxruntime_genai module surface.

    Covers the API surface used by LMMSORTGenAIEvaluator's __init__ + runtime paths.
    """
    return SimpleNamespace(
        Model=lambda *a, **kw: model,
        Tokenizer=lambda m: model.tokenizer,
        Generator=_FakeGenerator,
        GeneratorParams=_FakeGeneratorParams,
        Images=SimpleNamespace(open=lambda *paths: ("IMG", list(paths))),
        Audios=SimpleNamespace(open=lambda *paths: ("AUDIO", list(paths))),
        Config=lambda *a, **kw: SimpleNamespace(
            clear_providers=lambda: None,
            append_provider=lambda *a, **kw: None,
            set_provider_option=lambda *a, **kw: None,
        ),
    )


def _build_lmms_ortgenai_evaluator(tmp_path, fake_model):
    """Construct an LMMSORTGenAIEvaluator wired to a fake onnxruntime_genai.

    Returns ``(evaluator, og_patcher)``. The patcher is a context manager that
    swaps in the fake ``og`` module; tests should call evaluator methods inside
    ``with og_patcher: ...`` so runtime paths (``_score_continuation``,
    ``_run_generation``) also use the fake instead of the real ORT-GenAI.
    """
    from olive.evaluator import lmms_ort as lmms_ort_mod

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "genai_config.json").write_text('{"model": {"type": "phi4mm"}}', encoding="utf-8")

    fake_og = _make_fake_og(fake_model)
    _FakeGenerator.instances = []
    og_patcher = patch.object(lmms_ort_mod, "og", fake_og)
    with og_patcher:
        evaluator = LMMSORTGenAIEvaluator(
            pretrained=str(model_dir),
            batch_size=1,
            max_new_tokens=8,
            max_length=64,
            execution_provider="cpu",
            fail_on_error=True,
        )
    # Return a fresh patcher for the test to use during runtime calls.
    return evaluator, patch.object(lmms_ort_mod, "og", fake_og)


def test_score_continuation_uses_joint_tokenization_to_slice_continuation(tmp_path):
    """Encoding ``continuation`` standalone is wrong for sentencepiece/BPE tokenizers.

    The adapter must encode ``prompt + continuation`` jointly and slice off the
    prompt-aligned prefix so the scored tokens are the ones the model would
    actually produce extending the prompt. Verify by giving the tokenizer
    DIFFERENT tokens for ``continuation`` vs the prompt-suffix of
    ``prompt + continuation``: only the latter should land in append_tokens.
    """
    prompt = "<|user|>What is in the image?<|end|><|assistant|>"
    continuation = "A"

    fake_tokenizer = _FakeTokenizer(
        {
            prompt: [1, 2, 3, 4],
            prompt + continuation: [1, 2, 3, 4, 17, 18],  # cont tokens = [17, 18]
            continuation: [99999],  # standalone-encoded - MUST NOT be used
        }
    )

    # Three forward passes: one prompt-fill (generate_next_token) + two cont tokens.
    vocab_size = 50
    logits_prompt_end = np.full(vocab_size, -10.0, dtype=np.float32)
    logits_prompt_end[17] = 5.0  # greedy = 17
    logits_after_17 = np.full(vocab_size, -10.0, dtype=np.float32)
    logits_after_17[18] = 5.0  # greedy = 18
    logits_after_18 = np.full(vocab_size, -10.0, dtype=np.float32)

    fake_model = _FakeOgModel(
        tokenizer=fake_tokenizer,
        next_logits_queue=[logits_prompt_end, logits_after_17, logits_after_18],
        next_sampled_queue=[42],
    )

    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)

    with og_patcher:
        logprob, is_greedy = evaluator._score_continuation(prompt, continuation, images=[], audios=[])

    gen = _FakeGenerator.instances[-1]
    set_inputs_idx = next(i for i, c in enumerate(gen.calls) if c[0] == "set_inputs")
    next_token_idx = next(i for i, c in enumerate(gen.calls) if c[0] == "generate_next_token")
    append_calls = [c for c in gen.calls if c[0] == "append_tokens"]

    # set_inputs runs BEFORE the prompt-fill forward pass.
    assert set_inputs_idx < next_token_idx
    # The throwaway sample is rewound after the first iteration, before the first
    # real continuation token is appended.
    rewind_calls = [c for c in gen.calls if c[0] == "rewind_to"]
    assert len(rewind_calls) == 1
    # cont_tokens were correctly sliced from prompt+continuation, NOT taken from
    # encode(continuation) standalone (which would have been [99999]).
    assert [tok for _, tok in append_calls] == [[17], [18]]
    # Both predicted tokens were greedy (== argmax of their position's logits).
    assert is_greedy is True
    assert logprob < 0.0  # softmax(logits)[tok] is a probability in (0, 1) -> log negative


def test_score_continuation_triggers_forward_pass_before_first_get_logits(tmp_path):
    """Trigger compute after ``set_inputs`` before reading ``get_logits()``.

    ``set_inputs()`` does not run the decoder forward pass. The adapter must
    explicitly trigger it (``generate_next_token``) before the first
    ``get_logits()`` call, or that read returns undefined data.
    """
    prompt = "<|user|>x<|end|><|assistant|>"
    continuation = "y"
    fake_tokenizer = _FakeTokenizer({prompt: [1, 2], prompt + continuation: [1, 2, 5], continuation: [777]})
    logits = np.zeros(10, dtype=np.float32)
    logits[5] = 1.0
    fake_model = _FakeOgModel(
        tokenizer=fake_tokenizer,
        next_logits_queue=[logits, np.zeros(10, dtype=np.float32)],
        next_sampled_queue=[0],
    )

    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)
    with og_patcher:
        evaluator._score_continuation(prompt, continuation, images=[], audios=[])

    gen = _FakeGenerator.instances[-1]
    # The order must be: set_inputs, generate_next_token (prompt-fill compute),
    # then the loop's append_tokens. get_logits is read implicitly between
    # generate_next_token and the first append_tokens.
    op_names = [c[0] for c in gen.calls]
    set_inputs_idx = op_names.index("set_inputs")
    next_token_idx = op_names.index("generate_next_token")
    first_append_idx = op_names.index("append_tokens")
    assert set_inputs_idx < next_token_idx < first_append_idx


def test_score_continuation_returns_zero_when_continuation_tokenizes_to_empty_suffix(tmp_path):
    """Short-circuit cleanly when continuation contributes no tokens.

    Happens when prompt+cont tokenizes to the same length as prompt (e.g. cont
    is just whitespace absorbed by tokenizer normalization). The adapter must
    short-circuit instead of feeding an empty cont_tokens list to the loop.
    """
    prompt = "<|user|>x<|end|><|assistant|>"
    continuation = ""
    fake_tokenizer = _FakeTokenizer({prompt: [1, 2, 3], prompt + continuation: [1, 2, 3]})
    fake_model = _FakeOgModel(tokenizer=fake_tokenizer)

    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)
    with og_patcher:
        logprob, is_greedy = evaluator._score_continuation(prompt, continuation, images=[], audios=[])

    assert (logprob, is_greedy) == (0.0, True)
    # No generator should have been constructed for an empty continuation.
    assert not _FakeGenerator.instances


def test_run_generation_stops_on_eos_token(tmp_path):
    """Verify ``_run_generation`` stops at EOS tokens.

    ``_run_generation`` must respect ``_eos_token_ids`` and stop emitting text
    once the model samples an EOS token, even before max_new_tokens is reached.
    """
    fake_tokenizer = _FakeTokenizer({}, eos_token_ids=[99])
    logits = np.zeros(100, dtype=np.float32)
    fake_model = _FakeOgModel(
        tokenizer=fake_tokenizer,
        next_logits_queue=[logits] * 5,
        next_sampled_queue=[7, 8, 99, 10, 11],  # third sample is EOS
    )

    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)
    with og_patcher:
        out = evaluator._run_generation("p", images=[], audios=[], max_new_tokens=5)

    # Stream produces "<t{tok}>" per non-EOS token; EOS stops generation.
    assert out == "<t7><t8>"
    gen = _FakeGenerator.instances[-1]
    op_names = [c[0] for c in gen.calls]
    # Exactly 3 generate_next_token calls happened (7, 8, then EOS stops loop).
    assert op_names.count("generate_next_token") == 3


def test_run_generation_stops_on_explicit_stop_string(tmp_path):
    """``stop_strings`` should truncate output as soon as a stop sequence appears."""
    fake_tokenizer = _FakeTokenizer({}, eos_token_ids=[])
    logits = np.zeros(10, dtype=np.float32)
    fake_model = _FakeOgModel(
        tokenizer=fake_tokenizer,
        next_logits_queue=[logits] * 4,
        next_sampled_queue=[1, 2, 3, 4],
    )

    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)
    # _FakeTokenStream emits "<t1><t2><t3>..."; "<t2>" appears after the 2nd token.
    with og_patcher:
        out = evaluator._run_generation("p", images=[], audios=[], max_new_tokens=10, stop_strings=["<t2>"])

    # Output is truncated at (but not including) the stop string.
    assert out == "<t1>"
