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


def _packaged_files(package_data: dict[str, list[str]]) -> set[Path]:
    # setuptools resolves package_data globs relative to each package directory, so do the same here.
    # Path.match would also accept patterns rooted at the wrong package, which the wheel would miss.
    packaged = set()
    for package, patterns in package_data.items():
        package_dir = REPO_ROOT.joinpath(*package.split("."))
        for pattern in patterns:
            packaged.update(path.resolve() for path in package_dir.glob(pattern) if path.is_file())
    return packaged


def test_package_data_includes_all_runtime_data_files():
    packaged = _packaged_files(_get_package_data())
    not_packaged = {path.resolve() for pattern in NOT_PACKAGED for path in PACKAGE_ROOT.glob(pattern)}
    missing = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in _data_files()
        if path.resolve() not in packaged and path.resolve() not in not_packaged
    ]
    assert not missing, f"Data files missing from package_data in setup.py: {missing}"


def test_packaged_files_resolves_patterns_relative_to_the_package():
    # a pattern declared under the wrong package must not count as covering the file
    dockerfile = PACKAGE_ROOT / "engine" / "packaging" / "Dockerfile.base"
    assert dockerfile.resolve() not in _packaged_files({"olive": ["Dockerfile.base"]})
    assert dockerfile.resolve() in _packaged_files({"olive.engine.packaging": ["Dockerfile.base"]})
