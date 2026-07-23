# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest

from olive.common.quant.patterns import _assert_regex_safe, is_regex_pattern, match_override, match_skip


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

    def test_first_matching_pattern_wins_in_insertion_order(self):
        # both match — the first in config (insertion) order wins, regardless of length
        patterns = ["re:.*\\.experts\\..*", "re:layers\\.\\d+\\.experts\\.gate_proj"]
        match = match_override("layers.0.experts.gate_proj", patterns)
        assert match == "re:.*\\.experts\\..*"
        # reversing the order flips the winner
        assert match_override("layers.0.experts.gate_proj", list(reversed(patterns))) == (
            "re:layers\\.\\d+\\.experts\\.gate_proj"
        )

    def test_tie_break_is_deterministic(self):
        # same-length patterns: first (and here only) match wins
        patterns = ["re:foo.bar", "re:foo.baz"]
        # only the first matches
        assert match_override("foo.bar", patterns) == "re:foo.bar"

    def test_first_match_wins_even_when_later_pattern_is_more_specific(self):
        # A broad ``re:.*`` placed first wins over a later literal, per insertion order.
        patterns = ["re:.*", "model.embed_tokens"]
        assert match_override("model.embed_tokens", patterns) == "re:.*"
        # But a literal placed first wins over a later broad regex.
        assert match_override("model.embed_tokens", ["model.embed_tokens", "re:.*"]) == "model.embed_tokens"


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


class TestRegexSafetyBound:
    """Design item 8: ``re:`` patterns must have a real adversarial-input safety bound."""

    @pytest.mark.parametrize(
        "adversarial",
        [
            "(a+)+$",
            "(a*)*$",
            "(a+)*$",
            "(a*)+$",
            "(a|a)+$",
            "(a|aa)+$",
            "([a-z]+)+$",
            "(x+x+)+y",
            "(.*)*$",
            "(a+){2,}",
        ],
    )
    def test_rejects_nested_unbounded_quantifiers(self, adversarial):
        with pytest.raises(ValueError, match="nested unbounded quantifier"):
            _assert_regex_safe(adversarial)

    def test_rejects_overlong_pattern(self):
        with pytest.raises(ValueError, match="too long"):
            _assert_regex_safe("a" * 500)

    @pytest.mark.parametrize(
        "safe",
        [
            "layers\\.\\d+\\.experts\\.gate_proj",
            ".*\\.experts\\..*",
            "model\\.layers\\.\\d+\\.mlp\\.(gate|up|down)_proj",
            "(abc)+def",  # quantified group but body has no unbounded quantifier
            "a{2,4}b+",
            "[a-z]+\\.[0-9]+",
        ],
    )
    def test_accepts_safe_patterns(self, safe):
        # Should not raise, and should compile / match through the public API.
        _assert_regex_safe(safe)

    def test_adversarial_pattern_rejected_through_match_apis(self):
        # The safety bound is enforced when the pattern is actually used, not only when
        # validated directly.
        with pytest.raises(ValueError, match="nested unbounded quantifier"):
            match_override("aaaaaaaaaaaaaaaaaaaa!", ["re:(a+)+$"])
        with pytest.raises(ValueError, match="nested unbounded quantifier"):
            match_skip("aaaaaaaaaaaaaaaaaaaa!", ["re:(a+)+$"])
