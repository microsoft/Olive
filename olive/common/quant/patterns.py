# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pattern matching helpers for Olive quantization module selection.

`overrides` keys and `modules_to_not_convert` entries use the following
semantics:

- Plain string keys are *literal* for ``overrides`` (equality match) and
  *substring* for ``modules_to_not_convert`` (matches the existing HF
  ``modules_to_not_convert`` semantics).
- Keys prefixed with ``re:`` opt into regular-expression matching via
  ``re.fullmatch``.

When multiple ``overrides`` keys match the same target, the **first**
matching key in config (insertion) order wins (``match_override``); this
is the finalized precedence rule — "longest / most specific pattern" is
deliberately not used.

``re:`` patterns get a best-effort, compile-time UX check for
catastrophic-backtracking *shapes* (``_assert_regex_safe``): over-long
patterns and groups that nest any repetition/alternation (at any nesting
depth) inside an unbounded outer quantifier (e.g. ``(a+)+``,
``(a{1,2})+``, ``((a|aa))+$``) are rejected with a clear error before
matching is attempted.

This is **not a security boundary**. ``re:`` override/skip patterns come
from the user's own quantization config and run in the user's own
process — there is no trust boundary being crossed, so the scan is
intentionally a blacklist of known-bad shapes rather than a sound static
analysis. It catches common accidental catastrophic-backtracking shapes
early, with an actionable error, before they hang the process; other
exponential shapes (e.g. backreferences, adjacent unbounded quantifiers
like ``a*a*x$``) are not modeled and could still hang a user's own run.

These helpers are the single source of truth for the matching logic and
are used by both ``OliveHfQuantizationConfig`` and the Olive walker.
"""

from __future__ import annotations

import re
from functools import lru_cache

REGEX_PREFIX = "re:"

# Safety bound for user-supplied ``re:`` patterns. ``re.fullmatch`` has no timeout and a
# length cap alone does not prevent catastrophic backtracking (short adversarial patterns
# such as ``(a+)+$`` can already hang matching). We reject patterns that combine a group
# quantified with an unbounded quantifier (``*`` / ``+`` / ``{n,}``) with a body that itself
# contains *any* repetition (bounded or unbounded) or *any* alternation at *any* nesting
# depth or a top-level alternation — the classic ReDoS shapes ``(a+)+``, ``(a*)*``,
# ``(a|a)*``, the bounded-body variant ``(a{1,2})+``, and the nested-alternation variant
# ``((a|aa))+$`` — in addition to enforcing a conservative length cap.
#
# This is a best-effort, incomplete blacklist scan, not a soundness guarantee: it catches
# the shapes above early with a clear compile-time error, but does not model every possible
# exponential-backtracking construct (e.g. backreferences, adjacent unbounded quantifiers
# like ``a*a*x$``). ``re:`` patterns are trusted, user-authored config — not adversarial
# input — so this trade-off is intentional; see the module docstring.
_MAX_REGEX_LEN = 200


def is_regex_pattern(pattern: str) -> bool:
    """Return True if ``pattern`` opts into regex matching."""
    return isinstance(pattern, str) and pattern.startswith(REGEX_PREFIX)


def _skip_char_class(raw: str, i: int) -> int:
    """Return the index just past a ``[...]`` character class starting at ``raw[i] == '['``."""
    n = len(raw)
    j = i + 1
    if j < n and raw[j] == "^":
        j += 1
    if j < n and raw[j] == "]":  # a literal ``]`` as the first class member
        j += 1
    while j < n and raw[j] != "]":
        if raw[j] == "\\":
            j += 1
        j += 1
    return j + 1


def _brace_is_unbounded(raw: str, i: int) -> bool:
    """Whether the ``{...}`` quantifier starting at ``raw[i] == '{'`` has no finite upper bound."""
    close = raw.find("}", i)
    if close == -1:
        return False
    inner = raw[i + 1 : close]
    # ``{n,}`` (no upper bound) is unbounded; ``{n}`` / ``{n,m}`` are bounded.
    return inner.endswith(",") or ("," in inner and inner.split(",", 1)[1] == "")


def _body_has_repetition_or_alt(body: str) -> bool:
    """Whether a group body contains any repetition (bounded or unbounded) or alternation.

    A quantified group (``(...)+``, ``(...)*``, ``(...){n,}``) whose body itself repeats -
    whether or not that inner repetition is bounded - can still exhibit exponential
    backtracking (e.g. ``(a{1,2})+$`` on a run of "a"s), because the outer unbounded
    quantifier can match the same substring in exponentially many ways via the inner
    repetition's bounded-but-plural options. Alternation nested at *any* depth is just as
    dangerous (e.g. ``((a|aa))+$`` — the ``|`` is inside an inner group, not at the outer
    body's top level, but still gives the outer ``+`` exponentially many ways to partition
    a matched run). So *any* repetition or alternation anywhere inside a group that is
    itself quantified unboundedly is treated as unsafe — the ``depth`` counter below exists
    solely to correctly skip over character-class (``[...]``) contents, not to scope where
    ``|`` is considered unsafe.

    This is a best-effort blacklist scan (see module docstring) — not a claim that every
    accepted pattern is polynomial-time, only that this specific family of shapes is
    rejected.
    """
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            i = _skip_char_class(body, i)
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "|" or c in "*+?" or c == "{":
            return True
        i += 1
    return False


def _assert_regex_safe(raw: str) -> None:
    """Reject regex patterns matching known catastrophic-backtracking (ReDoS) shapes.

    Raises ``ValueError`` for over-long patterns and for nested unbounded quantifiers
    (repetition or alternation at any depth inside a group that is itself quantified
    unboundedly). This is a best-effort, incomplete static check, not a soundness
    guarantee — it catches common accidental catastrophic-backtracking shapes early, with
    a clear error, before they hang the process. It does not prove every accepted pattern
    is safe: other exponential shapes (backreferences, adjacent unbounded quantifiers like
    ``a*a*x$``) are not modeled. ``re:`` patterns are trusted, user-authored quantization
    config, not adversarial input, so this trade-off is intentional (see module docstring).
    """
    if len(raw) > _MAX_REGEX_LEN:
        raise ValueError(
            f"Regex override/skip pattern is too long ({len(raw)} > {_MAX_REGEX_LEN} chars); "
            "keep patterns short to bound matching cost."
        )
    stack: list[int] = []
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c == "\\":
            i += 2
            continue
        if c == "[":
            i = _skip_char_class(raw, i)
            continue
        if c == "(":
            stack.append(i)
        elif c == ")":
            start = stack.pop() if stack else -1
            nxt = raw[i + 1] if i + 1 < n else ""
            quantified_unbounded = nxt in ("*", "+") or (nxt == "{" and _brace_is_unbounded(raw, i + 1))
            if quantified_unbounded and start != -1 and _body_has_repetition_or_alt(raw[start + 1 : i]):
                raise ValueError(
                    f"Regex override/skip pattern {raw!r} contains a nested quantified group "
                    "(catastrophic-backtracking risk), e.g. `(a+)+` or `(a{1,2})+`. Rewrite the "
                    "pattern without a repeated/alternated group nested inside an unbounded "
                    "quantifier."
                )
        i += 1


@lru_cache(maxsize=512)
def _compiled(pattern: str) -> re.Pattern:
    raw = pattern[len(REGEX_PREFIX) :]
    _assert_regex_safe(raw)
    return re.compile(raw)


def match_override(name: str, patterns) -> str | None:
    """Find the override pattern for ``name`` using first-match-wins semantics.

    Patterns are evaluated in insertion order (config order); the **first** matching pattern
    wins. Plain string patterns match by **literal equality**; ``re:`` patterns match by
    ``re.fullmatch``. Insertion order (not "longest / most specific pattern") is the finalized,
    documented precedence rule so that overlapping overrides resolve deterministically.

    Returns the original pattern (with prefix preserved) so the caller can look it up in the
    underlying overrides dict, or ``None`` if no pattern matched.
    """
    if not patterns:
        return None
    for pattern in patterns:
        if is_regex_pattern(pattern):
            if _compiled(pattern).fullmatch(name):
                return pattern
        elif name == pattern:
            return pattern
    return None


def match_skip(name: str, patterns) -> bool:
    """Return True if ``name`` is matched by any skip pattern.

    Plain string patterns use **substring** matching to preserve the
    existing HF ``modules_to_not_convert`` semantics. ``re:`` patterns
    use ``re.fullmatch``.
    """
    if not patterns:
        return False
    for pattern in patterns:
        if is_regex_pattern(pattern):
            if _compiled(pattern).fullmatch(name):
                return True
        elif pattern and pattern in name:
            return True
    return False
