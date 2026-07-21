# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.common.quant.patterns import is_regex_pattern, match_override, match_skip


class TestIsRegexPattern:
    def test_returns_true_for_re_prefix(self):
        assert is_regex_pattern("re:foo")

    def test_returns_false_for_plain_string(self):
        assert not is_regex_pattern("model.embed_tokens")

    def test_returns_false_for_non_string(self):
        assert not is_regex_pattern(None)  # type: ignore[arg-type]


class TestMatchOverride:
    def test_returns_none_when_no_patterns(self):
        assert match_override("foo", []) is None
        assert match_override("foo", None) is None

    def test_literal_equality(self):
        assert match_override("model.embed_tokens", ["model.embed_tokens"]) == "model.embed_tokens"
        assert match_override("model.embed", ["model.embed_tokens"]) is None

    def test_regex_fullmatch(self):
        assert match_override("layers.0.experts.gate_proj", ["re:layers\\.\\d+\\.experts\\..*"]) == (
            "re:layers\\.\\d+\\.experts\\..*"
        )
        # not a fullmatch
        assert match_override("prefix_layers.0.experts", ["re:layers\\.\\d+\\.experts"]) is None

    def test_longest_pattern_wins(self):
        # both match — the more specific (longer) wins
        patterns = ["re:.*\\.experts\\..*", "re:layers\\.\\d+\\.experts\\.gate_proj"]
        match = match_override("layers.0.experts.gate_proj", patterns)
        assert match == "re:layers\\.\\d+\\.experts\\.gate_proj"

    def test_tie_break_is_deterministic(self):
        # same-length patterns: lexically smallest wins
        patterns = ["re:foo.bar", "re:foo.baz"]
        # only the first matches
        assert match_override("foo.bar", patterns) == "re:foo.bar"

    def test_literal_beats_shorter_regex_when_longer(self):
        patterns = ["re:.*", "model.embed_tokens"]
        # literal pattern (len 18) is longer than "re:.*" (len 5)
        assert match_override("model.embed_tokens", patterns) == "model.embed_tokens"


class TestMatchSkip:
    def test_substring_match_for_plain_string(self):
        # substring preserves HF semantics
        assert match_skip("model.layers.0.experts.gate_proj", ["experts"])
        assert match_skip("model.embed_tokens", ["embed_tokens"])
        assert not match_skip("model.layers.0.attn.q_proj", ["experts"])

    def test_regex_fullmatch_for_re_prefix(self):
        assert match_skip("model.layers.0.experts.router.gate", ["re:.*\\.router\\.gate"])
        assert not match_skip("model.layers.0.experts.router.gate.bias", ["re:.*\\.router\\.gate"])

    def test_empty_pattern_does_not_match(self):
        assert not match_skip("foo", [""])

    def test_none_or_empty_patterns(self):
        assert not match_skip("foo", None)
        assert not match_skip("foo", [])

    @pytest.mark.parametrize(
        ("name", "patterns", "expected"),
        [
            ("model.layers.0.experts.0.w1", ["experts"], True),
            ("model.layers.0.attn.q_proj", ["experts"], False),
            ("router.gate", ["re:router\\.gate"], True),
            ("router_gate", ["re:router\\.gate"], False),
        ],
    )
    def test_parametrized(self, name, patterns, expected):
        assert match_skip(name, patterns) is expected
