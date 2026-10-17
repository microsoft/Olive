# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "olive"

# data files that are only referenced from the docs as examples and are not loaded at runtime
NOT_PACKAGED = {"assets/cost_models/*.csv"}


def _get_package_data() -> dict[str, list[str]]:
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "package_data":
            return ast.literal_eval(node.value)
    raise AssertionError("package_data not found in setup.py")


def _data_files() -> list[Path]:
    return [
        path
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix not in {".py", ".pyc"} and "__pycache__" not in path.parts
    ]


def test_package_data_entries_point_to_existing_packages():
    for package in _get_package_data():
        package_dir = REPO_ROOT.joinpath(*package.split("."))
        assert (package_dir / "__init__.py").exists(), f"package_data entry {package} is not a package"


def test_package_data_includes_all_runtime_data_files():
    package_data = _get_package_data()
    missing = []
    for path in _data_files():
        relative_path = path.relative_to(PACKAGE_ROOT)
        if any(relative_path.match(pattern) for pattern in NOT_PACKAGED):
            continue
        covered = False
        for package, patterns in package_data.items():
            package_dir = REPO_ROOT.joinpath(*package.split("."))
            if package_dir in path.parents and any(
                path.relative_to(package_dir).match(pattern) for pattern in patterns
            ):
                covered = True
                break
        if not covered:
            missing.append(str(relative_path))
    assert not missing, f"Data files missing from package_data in setup.py: {missing}"
