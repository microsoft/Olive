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

These helpers are the single source of truth for the matching logic and
are used by both ``OliveHfQuantizationConfig`` and the Olive walker.
"""

from __future__ import annotations

import re
from functools import cache

REGEX_PREFIX = "re:"


def is_regex_pattern(pattern: str) -> bool:
    """Return True if ``pattern`` opts into regex matching."""
    return isinstance(pattern, str) and pattern.startswith(REGEX_PREFIX)


@cache
def _compiled(pattern: str) -> re.Pattern:
    return re.compile(pattern[len(REGEX_PREFIX) :])


def match_override(name: str, patterns) -> str | None:
    """Find the best override pattern for ``name``.

    Plain string patterns match by **literal equality**. ``re:`` patterns
    match by ``re.fullmatch``. The most specific match (longest pattern
    string) wins; ties are broken lexicographically so the choice is
    deterministic.

    Returns the original pattern (with prefix preserved) so the caller
    can look it up in the underlying overrides dict, or ``None`` if no
    pattern matched.
    """
    if not patterns:
        return None
    matches: list[str] = []
    for pattern in patterns:
        if is_regex_pattern(pattern):
            if _compiled(pattern).fullmatch(name):
                matches.append(pattern)
        elif name == pattern:
            matches.append(pattern)
    if not matches:
        return None
    # longest pattern first, then lexical
    return sorted(matches, key=lambda p: (-len(p), p))[0]


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
