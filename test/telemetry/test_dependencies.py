# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Guard against olive.telemetry importing packages that are not declared install dependencies.

``olive.telemetry`` is imported by ``import olive`` itself, so every third-party module it
imports unconditionally must be listed in ``requirements.txt``. Otherwise a plain
``pip install olive-ai`` (no extras) is broken at import time.
"""

import ast
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_ROOT = REPO_ROOT / "olive" / "telemetry"
REQUIREMENTS_TXT = REPO_ROOT / "requirements.txt"


def _normalize(name: str) -> str:
    """Normalize a distribution name as described in PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_requirements() -> set[str]:
    declared = set()
    for raw_line in REQUIREMENTS_TXT.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        if match:
            declared.add(_normalize(match.group(0)))
    return declared


def _unconditional_top_level_imports(path: Path) -> set[str]:
    """Return the top-level package names a module imports at module scope without any guard.

    Only statements directly in the module body count: imports inside ``try``/``except``,
    ``if TYPE_CHECKING:`` or functions are not hard dependencies.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _third_party_imports() -> dict[str, set[str]]:
    """Map each third-party top-level module to the telemetry files that import it."""
    imports: dict[str, set[str]] = {}
    for path in sorted(TELEMETRY_ROOT.rglob("*.py")):
        for name in _unconditional_top_level_imports(path):
            if name == "olive" or name in sys.stdlib_module_names:
                continue
            imports.setdefault(name, set()).add(path.relative_to(REPO_ROOT).as_posix())
    return imports


@pytest.mark.skipif(not REQUIREMENTS_TXT.exists(), reason="requires a source checkout of the repository")
def test_telemetry_imports_are_declared_in_requirements():
    declared = _declared_requirements()
    distributions = packages_distributions()

    undeclared = {}
    for module, files in _third_party_imports().items():
        candidates = {_normalize(dist) for dist in distributions.get(module, [module])}
        if not candidates & declared:
            undeclared[module] = sorted(files)

    assert not undeclared, (
        "olive.telemetry is imported by `import olive`, but these modules are not declared in "
        f"requirements.txt: {undeclared}"
    )
