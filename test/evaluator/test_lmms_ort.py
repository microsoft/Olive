# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Tests intentionally exercise "protected" runtime methods (_score_continuation,
# _run_generation) and configure fake collaborators by setting attributes
# directly on the fake. Both are normal in unit tests, so suppress pylint's
# protected-access / attribute-defined-outside-init warnings for this file.
# pylint: disable=protected-access,attribute-defined-outside-init,no-value-for-parameter,unexpected-keyword-arg
import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest

from olive.evaluator.lmms_ort import (
    LMMSORTGenAIEvaluator,
    _build_whisper_prompt,
    _normalize_execution_provider,
)
from olive.evaluator.olive_evaluator import LMMSEvaluator
from olive.model import ONNXModelHandler


def test_build_whisper_prompt_configures_language_and_task():
    prompt = _build_whisper_prompt(
        whisper_language="fr",
        whisper_task="translate",
        whisper_timestamps=False,
    )

    assert prompt == "<|startoftranscript|><|fr|><|translate|><|notimestamps|>"


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen2_5_vl", "You are a helpful assistant."),
        ("qwen2_5_vl_text", "You are a helpful assistant."),
        ("qwen3_vl", "You are a helpful assistant."),
        ("gemma4", "You are a helpful assistant."),
        ("gemma3", "You are a helpful assistant."),
        (
            "qwen2_5_omni",
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        ),
        (
            "qwen3_omni_moe",
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        ),
        ("phi4mm", ""),
        ("phi3v", ""),
        ("whisper", "You are a helpful assistant."),
        (None, "You are a helpful assistant."),
    ],
)
def test_default_system_prompt_for_model_type_matches_lmms_eval_wrappers(model_type, expected):
    assert LMMSORTGenAIEvaluator.default_system_prompt_for_model_type(model_type) == expected


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
    inst._supports_structured_content = True
    inst.whisper_language = "en"
    inst.whisper_task = "transcribe"
    inst.whisper_timestamps = False
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
    assert messages[1] == {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "What is in the image?"}],
    }


def test_build_prompt_for_request_skips_system_when_empty():
    inst = _make_evaluator_for_prompt_tests()
    inst.system_prompt = ""
    inst._tokenizer.apply_chat_template.return_value = "out"

    inst._build_prompt_for_request("Q", num_images=0, num_audios=0)

    import json as _json

    messages = _json.loads(inst._tokenizer.apply_chat_template.call_args.args[0])
    assert all(m["role"] != "system" for m in messages)
    assert messages[-1] == {"role": "user", "content": "Q"}


def test_build_prompt_for_request_uses_structured_media_parts():
    inst = _make_evaluator_for_prompt_tests()
    inst.system_prompt = ""
    inst._tokenizer.apply_chat_template.return_value = "out"

    inst._build_prompt_for_request("Q", num_images=2, num_audios=1)

    import json as _json

    messages = _json.loads(inst._tokenizer.apply_chat_template.call_args.args[0])
    assert messages[0]["content"] == [
        {"type": "image"},
        {"type": "image"},
        {"type": "audio"},
        {"type": "text", "text": "Q"},
    ]


def test_build_prompt_for_request_uses_structured_content_when_supported():
    """Use structured content parts when supported."""
    inst = _make_evaluator_for_prompt_tests()
    inst._tokenizer.apply_chat_template.return_value = "<rendered>"

    inst._build_prompt_for_request("What is this?", num_images=1, num_audios=1)

    import json as _json

    messages = _json.loads(inst._tokenizer.apply_chat_template.call_args.args[0])
    user_msg = messages[-1]
    assert user_msg["role"] == "user"
    # Content remains a list of typed parts rather than a flat string.
    assert user_msg["content"] == [
        {"type": "image"},
        {"type": "audio"},
        {"type": "text", "text": "What is this?"},
    ]


def test_build_prompt_for_request_rejects_media_when_structured_content_unsupported():
    inst = _make_evaluator_for_prompt_tests()
    inst._supports_structured_content = False

    with pytest.raises(ValueError, match="structured image/audio content"):
        inst._build_prompt_for_request("What is this?", num_images=1, num_audios=0)


def test_probe_structured_content_support_detects_injection():
    """A template that injects the media token (no dict repr) -> supported."""
    inst = LMMSORTGenAIEvaluator.__new__(LMMSORTGenAIEvaluator)
    inst._tokenizer = MagicMock()
    inst._tokenizer.apply_chat_template.side_effect = [
        "<|im_start|>user\n<|image|>x<|im_end|>",
        "<|im_start|>user\nx<|im_end|>",
    ]

    assert inst._probe_structured_content_support() is True


def test_probe_structured_content_support_detects_stringified_repr():
    """A template that leaks the Python dict repr -> not supported."""
    inst = LMMSORTGenAIEvaluator.__new__(LMMSORTGenAIEvaluator)
    inst._tokenizer = MagicMock()
    inst._tokenizer.apply_chat_template.return_value = "<|user|>[{'type': 'image'}, ...]<|end|>"

    assert inst._probe_structured_content_support() is False


def test_probe_structured_content_support_detects_ignored_image_part():
    inst = LMMSORTGenAIEvaluator.__new__(LMMSORTGenAIEvaluator)
    inst._tokenizer = MagicMock()
    inst._tokenizer.apply_chat_template.side_effect = ["same-rendering", "same-rendering"]

    assert inst._probe_structured_content_support() is False


def _make_evaluator_for_generate_until(ignore_stop_strings):
    """Construct an LMMSORTGenAIEvaluator with __init__ skipped.

    Wired so generate_until runs without a real model: _run_generation is a stub
    that records the stop list it was given.
    """
    inst = LMMSORTGenAIEvaluator.__new__(LMMSORTGenAIEvaluator)
    inst.max_new_tokens = 256
    inst.ignore_stop_strings = set(ignore_stop_strings or [])
    inst.cache_hook = SimpleNamespace(add_partial=lambda *a, **k: None)
    inst._get_doc_and_visuals = lambda *a, **k: (None, [])
    inst._build_prompt_for_request = lambda *a, **k: "<prompt>"
    inst.recorded_stops = []

    def _fake_run_generation(prompt, images, audios, max_new, stop, search_options=None):
        inst.recorded_stops.append(stop)
        inst.recorded_search_options = search_options
        return "ok"

    inst._run_generation = _fake_run_generation
    return inst


def _make_generate_until_request(until):
    gen_kwargs = {"until": until}
    return SimpleNamespace(args=("ctx", gen_kwargs, None, 0, "task", "split"))


def test_generate_until_drops_ignored_stop_strings():
    inst = _make_evaluator_for_generate_until(ignore_stop_strings=["\n\n"])
    req = _make_generate_until_request(until=["\n\n", "Q:"])

    inst.generate_until([req], disable_tqdm=True)

    # "\n\n" is suppressed, the legitimate "Q:" stop is preserved.
    assert inst.recorded_stops == [["Q:"]]


def test_generate_until_sets_stop_none_when_all_ignored():
    inst = _make_evaluator_for_generate_until(ignore_stop_strings=["\n\n"])
    req = _make_generate_until_request(until=["\n\n"])

    inst.generate_until([req], disable_tqdm=True)

    # The only stop string was suppressed -> None (generation runs to EOS).
    assert inst.recorded_stops == [None]


def test_generate_until_keeps_stops_when_nothing_to_ignore():
    inst = _make_evaluator_for_generate_until(ignore_stop_strings=[])
    req = _make_generate_until_request(until=["\n\n", "Q:"])

    inst.generate_until([req], disable_tqdm=True)

    # No ignore list configured -> stops pass through unchanged.
    assert inst.recorded_stops == [["\n\n", "Q:"]]


def test_generate_until_forwards_supported_search_options():
    inst = _make_evaluator_for_generate_until(ignore_stop_strings=[])
    req = SimpleNamespace(
        args=(
            "ctx",
            {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 10,
                "repetition_penalty": 1.1,
            },
            None,
            0,
            "task",
            "split",
        )
    )

    inst.generate_until([req], disable_tqdm=True)

    assert inst.recorded_search_options == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 10,
        "repetition_penalty": 1.1,
    }


def test_generate_until_rejects_unsupported_beam_search():
    inst = _make_evaluator_for_generate_until(ignore_stop_strings=[])
    req = SimpleNamespace(args=("ctx", {"num_beams": 2}, None, 0, "task", "split"))

    with pytest.raises(NotImplementedError, match="Beam search"):
        inst.generate_until([req], disable_tqdm=True)


def test_wrap_generate_until_drop_stops_filters_hf_wrapper_until():
    """Strip ignored stops before delegating to the HF wrapper.

    The patch updates each request's generation kwargs in place.
    """
    from olive.evaluator.olive_evaluator import _wrap_generate_until_drop_stops

    seen_until = []

    class FakeHfWrapper:
        def generate_until(self, requests, *args, **kwargs):
            seen_until.extend(r.args[1].get("until") for r in requests)
            return ["ok"] * len(requests)

    lm = FakeHfWrapper()
    _wrap_generate_until_drop_stops(lm, ["\n\n"])

    reqs = [
        _make_generate_until_request(until=["\n\n", "Q:"]),  # partial -> keep "Q:"
        _make_generate_until_request(until=["\n\n"]),  # all ignored -> None
        _make_generate_until_request(until="\n\n"),  # str form -> None
    ]
    out = lm.generate_until(reqs)

    assert out == ["ok", "ok", "ok"]
    assert seen_until == [["Q:"], [], []]


def test_wrap_generate_until_drop_stops_finds_chat_wrapper_kwargs():
    from olive.evaluator.olive_evaluator import _wrap_generate_until_drop_stops

    seen_until = []

    class FakeChatWrapper:
        def generate_until(self, requests, *args, **kwargs):
            seen_until.extend(request.args[2]["until"] for request in requests)
            return ["ok"] * len(requests)

    lm = FakeChatWrapper()
    _wrap_generate_until_drop_stops(lm, ["\n\n"])
    request = SimpleNamespace(args=("ctx", lambda doc: doc, {"until": ["\n\n", "END"]}, 0, "task", "split"))

    lm.generate_until([request])

    assert seen_until == [["END"]]


def test_wrap_generate_until_drop_stops_noop_without_ignore_list():
    from olive.evaluator.olive_evaluator import _wrap_generate_until_drop_stops

    class FakeHfWrapper:
        def generate_until(self, requests, *args, **kwargs):
            return "orig"

    lm = FakeHfWrapper()
    _wrap_generate_until_drop_stops(lm, [])
    # Empty ignore list -> generate_until not wrapped; original behavior intact.
    assert "generate_until" not in vars(lm)
    assert lm.generate_until([]) == "orig"


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
        system_prompt=None,
        execution_provider=["CUDAExecutionProvider"],
        provider_options=None,
        fail_on_error=False,
        ignore_stop_strings=None,
        whisper_language="en",
        whisper_task="transcribe",
        whisper_timestamps=False,
    )
    simple_evaluate_mock.assert_called_once()
    assert result.get_value("ai2d_lite", "exact_match") == 0.5
    assert output_path.exists()


def test_lmms_evaluator_preserves_filter_names_and_metric_direction(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "text.onnx"
    model_path.touch()
    (model_dir / "genai_config.json").write_text('{"model": {"type": "phi4mm"}}', encoding="utf-8")
    evaluator = LMMSEvaluator(tasks=["speech_task"])
    model = ONNXModelHandler(model_path=str(model_path))
    simple_evaluate_result = {
        "results": {
            "speech_task": {
                "alias": "Speech",
                "wer,strict": 0.2,
                "wer,flexible": 0.1,
                "wer_stderr,strict": 0.01,
            }
        },
        "higher_is_better": {"speech_task": {"wer": False}},
        "configs": {},
    }
    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = MagicMock(return_value=simple_evaluate_result)

    with (
        patch.dict(
            sys.modules,
            {"lmms_eval": lmms_eval_module, "lmms_eval.evaluator": lmms_eval_evaluator_module},
        ),
        patch("olive.evaluator.lmms_ort.LMMSORTGenAIEvaluator", return_value=SimpleNamespace()),
    ):
        result = evaluator.evaluate(model, [])

    assert result.root["speech_task-wer,strict"].value == 0.2
    assert result.root["speech_task-wer,flexible"].value == 0.1
    assert result.root["speech_task-wer,strict"].higher_is_better is False
    assert all("_stderr" not in key for key in result.root)


def test_lmms_output_path_avoids_cross_backend_overwrite(tmp_path):
    from olive.evaluator.olive_evaluator import _lmms_output_path

    output_path = tmp_path / "results.json"
    output_path.write_text('{"model_type": "hf"}', encoding="utf-8")
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    model = ONNXModelHandler(model_path=str(model_path))

    assert _lmms_output_path(str(output_path), model) == tmp_path / "results_onnx.json"


def test_lmms_model_size_counts_whole_genai_package(tmp_path):
    from olive.evaluator.olive_evaluator import _lmms_model_size_on_disk

    package = tmp_path / "package"
    decoder = package / "decoder"
    vision = package / "vision"
    decoder.mkdir(parents=True)
    vision.mkdir()
    (package / "genai_config.json").write_text("{}", encoding="utf-8")
    (decoder / "model.onnx").write_bytes(b"decoder")
    (vision / "model.onnx").write_bytes(b"vision")
    model = ONNXModelHandler(model_path=str(package), onnx_file_name="decoder/model.onnx")

    assert _lmms_model_size_on_disk(model) == sum(path.stat().st_size for path in package.rglob("*") if path.is_file())


def test_summarize_lmms_samples_counts_nested_mathvision_score():
    from olive.evaluator.olive_evaluator import _summarize_lmms_samples

    summary = _summarize_lmms_samples(
        "mathvision_test",
        [
            {
                "doc_id": 0,
                "target": "A",
                "resps": [["answer"]],
                "mathvision_standard_eval": {"scores": [True], "response": ["answer"]},
            }
        ],
    )

    assert summary["n_scored_correct"] == 1


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

    def __init__(
        self,
        pretrained,
        device="cuda",
        dtype="auto",
        batch_size=1,
        trust_remote_code=True,
        system_prompt=None,
        **kwargs,
    ):
        type(self).last_kwargs = {
            "pretrained": pretrained,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "trust_remote_code": trust_remote_code,
            "system_prompt": system_prompt,
            **kwargs,
        }


class _FakeQwenWrapper:
    """Fake lmms-eval wrapper with a qwen2_5_vl-style signature.

    Does NOT accept dtype or trust_remote_code (mirrors lmms-eval's qwen2_5_vl
    class which asserts kwargs == {}).
    """

    last_kwargs: ClassVar[dict] = {}

    def __init__(
        self,
        pretrained,
        device="cuda",
        device_map="auto",
        batch_size=1,
        system_prompt=None,
        **kwargs,
    ):
        if kwargs:
            raise AssertionError(f"Unexpected kwargs: {kwargs}")
        type(self).last_kwargs = {
            "pretrained": pretrained,
            "device": device,
            "device_map": device_map,
            "batch_size": batch_size,
            "system_prompt": system_prompt,
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


class _FakeWhisperWrapper:
    last_kwargs: ClassVar[dict] = {}

    def __init__(self, pretrained, device="cuda", batch_size=1, language="en", task="transcribe"):
        type(self).last_kwargs = {
            "pretrained": pretrained,
            "device": device,
            "batch_size": batch_size,
            "language": language,
            "task": task,
        }


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

    lmms_eval_models_module.get_model.assert_called_once_with("phi4_multimodal", force_simple=True)
    assert _FakePhi4Wrapper.last_kwargs["pretrained"] == "/local/path/Phi-4-multimodal-instruct"
    assert _FakePhi4Wrapper.last_kwargs["batch_size"] == 2
    # trust_remote_code defaults to False (see olive/evaluator/olive_evaluator.py
    # LMMSEvaluator.__init__); users opt in explicitly in the recipe.
    assert _FakePhi4Wrapper.last_kwargs["trust_remote_code"] is False
    assert _FakePhi4Wrapper.last_kwargs["dtype"] == "auto"
    assert _FakePhi4Wrapper.last_kwargs["system_prompt"] == ""
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
    assert _FakeQwenWrapper.last_kwargs["system_prompt"] == "You are a helpful assistant."


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

    lmms_eval_models_module.get_model.assert_called_once_with("qwen2_5_vl", force_simple=True)
    # Explicit model_class bypasses class auto-detection, but model_type is still
    # consulted to resolve the parity-correct default system prompt.
    handler_stub.get_hf_model_type.assert_called_once()


def test_lmms_evaluator_forwards_whisper_language_and_task(monkeypatch):
    handler_stub = _make_hf_model_handler_stub("/p/whisper", "whisper")
    _patch_isinstance_for_hf(handler_stub, monkeypatch)
    evaluator = LMMSEvaluator(
        tasks=["fleurs_fr"],
        whisper_language="fr",
        whisper_task="translate",
    )
    lmms_eval_module = ModuleType("lmms_eval")
    lmms_eval_evaluator_module = ModuleType("lmms_eval.evaluator")
    lmms_eval_evaluator_module.simple_evaluate = MagicMock(return_value={"results": {}, "configs": {}})
    lmms_eval_models_module = ModuleType("lmms_eval.models")
    lmms_eval_models_module.get_model = MagicMock(return_value=_FakeWhisperWrapper)

    with patch.dict(
        sys.modules,
        {
            "lmms_eval": lmms_eval_module,
            "lmms_eval.evaluator": lmms_eval_evaluator_module,
            "lmms_eval.models": lmms_eval_models_module,
        },
    ):
        evaluator.evaluate(handler_stub, [])

    assert _FakeWhisperWrapper.last_kwargs["language"] == "fr"
    assert _FakeWhisperWrapper.last_kwargs["task"] == "translate"


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


def test_partition_visuals_downmixes_channels_first_and_last_audio():
    from olive.evaluator.lmms_ort import _partition_visuals

    channels_first = {"array": np.ones((2, 16), dtype=np.float32), "sampling_rate": 16000}
    channels_last = {"array": np.ones((16, 2), dtype=np.float32), "sampling_rate": 16000}

    _, audios = _partition_visuals([channels_first, channels_last])

    assert [arr.shape for arr, _ in audios] == [(16,), (16,)]


def test_partition_visuals_rejects_video():
    from olive.evaluator.lmms_ort import _partition_visuals

    with pytest.raises(NotImplementedError, match="video"):
        _partition_visuals(["sample.mp4"])


def test_structured_prompt_preserves_placeholder_order():
    inst = _make_evaluator_for_prompt_tests()
    inst._tokenizer.apply_chat_template.return_value = "<rendered>"

    inst._build_prompt_for_request("before <image> middle <audio> after", num_images=1, num_audios=1)

    import json as _json

    messages = _json.loads(inst._tokenizer.apply_chat_template.call_args.args[0])
    assert messages[-1]["content"] == [
        {"type": "text", "text": "before "},
        {"type": "image"},
        {"type": "text", "text": " middle "},
        {"type": "audio"},
        {"type": "text", "text": " after"},
    ]


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

    def apply_chat_template(self, messages, add_generation_prompt=True):
        media_token = "<|image|>" if '"type": "image"' in messages else ""
        return f"{media_token}x"


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
        self.calls.append(("set_inputs", inputs))
        self._token_count = int(inputs["input_ids"].shape()[-1])
        # ORT-GenAI SetInputs appends prompt tokens and computes prompt logits.
        self._consume_forward_pass()

    def generate_next_token(self):
        if self._current_logits is None:
            self._consume_forward_pass()
        sampled = self._sampled_queue.pop(0) if self._sampled_queue else -1
        self._last_sampled = sampled
        self._token_count += 1
        self.calls.append(("generate_next_token", sampled))
        self._current_logits = None

    def get_logits(self):
        self.calls.append(("get_logits",))
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
        max_length = self._params.search_options.get("max_length")
        return self._done or (max_length is not None and self._token_count >= max_length)

    def rewind_to(self, n):
        self.calls.append(("rewind_to", n))
        self._token_count = n


class _FakeGeneratorParams:
    def __init__(self, model):
        self._model = model
        self.search_options = {}

    def set_search_options(self, **kwargs):
        self.search_options.update(kwargs)


class _FakeTensor:
    def __init__(self, shape):
        self._shape = shape

    def shape(self):
        return self._shape


class _FakeProcessor:
    def __init__(self, input_length):
        self._input_length = input_length

    def __call__(self, prompt, images=None, audios=None):
        return {
            "input_ids": _FakeTensor([1, self._input_length]),
            "_prompt": prompt,
            "_images": images,
            "_audios": audios,
        }


class _FakeOgModel:
    def __init__(self, tokenizer=None, next_logits_queue=None, next_sampled_queue=None, processed_input_length=4):
        self.tokenizer = tokenizer
        self._next_logits_queue = list(next_logits_queue or [])
        self._next_sampled_queue = list(next_sampled_queue or [])
        self.processed_input_length = processed_input_length

    def create_multimodal_processor(self):
        return _FakeProcessor(self.processed_input_length)


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

    # Two forward passes: prompt prefill in set_inputs + cont[0] append.
    vocab_size = 50
    logits_prompt_end = np.full(vocab_size, -10.0, dtype=np.float32)
    logits_prompt_end[17] = 5.0  # greedy = 17
    logits_after_17 = np.full(vocab_size, -10.0, dtype=np.float32)
    logits_after_17[18] = 5.0  # greedy = 18
    logits_after_18 = np.full(vocab_size, -10.0, dtype=np.float32)

    fake_model = _FakeOgModel(
        tokenizer=fake_tokenizer,
        next_logits_queue=[logits_prompt_end, logits_after_17, logits_after_18],
    )

    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)

    with og_patcher:
        loss, is_greedy = evaluator._score_continuation(prompt, continuation, images=[], audios=[])

    gen = _FakeGenerator.instances[-1]
    set_inputs_idx = next(i for i, c in enumerate(gen.calls) if c[0] == "set_inputs")
    append_calls = [c for c in gen.calls if c[0] == "append_tokens"]

    assert set_inputs_idx < next(i for i, c in enumerate(gen.calls) if c[0] == "get_logits")
    assert not [c for c in gen.calls if c[0] in ("generate_next_token", "rewind_to")]
    # cont_tokens were correctly sliced from prompt+continuation, NOT taken from
    # encode(continuation) standalone (which would have been [99999]).
    assert [tok for _, tok in append_calls] == [[17]]
    # Both predicted tokens were greedy (== argmax of their position's logits).
    assert is_greedy is True
    assert loss > 0.0
    assert gen._params.search_options["max_length"] == 6


def test_score_continuation_uses_prefill_logits_from_set_inputs(tmp_path):
    """Read prompt logits immediately after SetInputs without sampling a token."""
    prompt = "<|user|>x<|end|><|assistant|>"
    continuation = "y"
    fake_tokenizer = _FakeTokenizer({prompt: [1, 2], prompt + continuation: [1, 2, 5], continuation: [777]})
    logits = np.zeros(10, dtype=np.float32)
    logits[5] = 1.0
    fake_model = _FakeOgModel(
        tokenizer=fake_tokenizer,
        next_logits_queue=[logits],
    )

    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)
    with og_patcher:
        evaluator._score_continuation(prompt, continuation, images=[], audios=[])

    gen = _FakeGenerator.instances[-1]
    op_names = [c[0] for c in gen.calls]
    set_inputs_idx = op_names.index("set_inputs")
    get_logits_idx = op_names.index("get_logits")
    assert set_inputs_idx < get_logits_idx
    assert "generate_next_token" not in op_names
    assert "rewind_to" not in op_names


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


def test_score_continuation_rejects_target_beyond_context_ceiling(tmp_path):
    prompt = "<|user|>x<|end|><|assistant|>"
    continuation = "yz"
    fake_tokenizer = _FakeTokenizer(
        {
            prompt: [1, 2],
            prompt + continuation: [1, 2, 5, 6],
        }
    )
    fake_model = _FakeOgModel(tokenizer=fake_tokenizer, processed_input_length=63)
    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)

    with og_patcher, pytest.raises(RuntimeError, match="cannot be scored within max_length"):
        evaluator._score_continuation(prompt, continuation, images=[], audios=[])

    assert not _FakeGenerator.instances


def test_loglikelihood_accepts_string_continuation_from_multiple_choice():
    evaluator = LMMSORTGenAIEvaluator.__new__(LMMSORTGenAIEvaluator)
    evaluator.cache_hook = SimpleNamespace(add_partial=lambda *args, **kwargs: None)
    evaluator._get_doc_and_visuals = lambda *args, **kwargs: ({"answer": "A"}, [])
    evaluator._build_prompt_for_request = lambda *args, **kwargs: "<prompt>"
    evaluator._score_continuation = MagicMock(return_value=(0.25, True))
    request = SimpleNamespace(args=("context", " A", None, 0, "arc_easy", "test"))

    result = evaluator.loglikelihood([request], disable_tqdm=True)

    assert result == [(0.25, True)]
    evaluator._score_continuation.assert_called_once_with("<prompt>", " A", [], [])


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
    assert gen._params.search_options["max_length"] == 9


def test_run_generation_clamps_budget_to_remaining_context(tmp_path):
    fake_tokenizer = _FakeTokenizer({}, eos_token_ids=[])
    logits = np.zeros(10, dtype=np.float32)
    fake_model = _FakeOgModel(
        tokenizer=fake_tokenizer,
        next_logits_queue=[logits] * 2,
        next_sampled_queue=[7, 8, 9],
        processed_input_length=62,
    )
    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)

    with og_patcher, patch("olive.evaluator.lmms_ort.logger.warning") as warning_mock:
        output = evaluator._run_generation("p", images=[], audios=[], max_new_tokens=5)

    assert output == "<t7><t8>"
    generator = _FakeGenerator.instances[-1]
    assert generator._params.search_options["max_length"] == 64
    warning_mock.assert_called_once_with(
        "Reducing max_new_tokens from %d to %d because the processed prompt uses %d of max_length=%d.",
        5,
        2,
        62,
        64,
    )


def test_run_generation_rejects_prompt_at_context_ceiling(tmp_path):
    fake_model = _FakeOgModel(
        tokenizer=_FakeTokenizer({}, eos_token_ids=[]),
        processed_input_length=64,
    )
    evaluator, og_patcher = _build_lmms_ortgenai_evaluator(tmp_path, fake_model)

    with og_patcher, pytest.raises(RuntimeError, match="processed prompt exceeds max_length"):
        evaluator._run_generation("p", images=[], audios=[], max_new_tokens=1)

    assert not _FakeGenerator.instances


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
