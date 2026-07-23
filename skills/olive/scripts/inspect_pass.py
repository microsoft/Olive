#!/usr/bin/env python
# ruff: noqa: T201

import argparse
import json
from pathlib import Path
from typing import Any

from olive.hardware import AcceleratorSpec
from olive.package_config import OlivePackageConfig


def _serialize(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if hasattr(value, "to_json"):
        return _serialize(value.to_json())
    return str(value)


def _type_name(annotation: Any) -> str:
    if annotation is None:
        return "Any"
    if getattr(annotation, "__module__", None) == "builtins" and (name := getattr(annotation, "__name__", None)):
        return name
    return str(annotation).replace("typing.", "")


def _matches(values: set[str], requested: str | None) -> bool:
    return requested is None or "*" in values or requested in values


def _list_passes(package_config: OlivePackageConfig, args: argparse.Namespace):
    passes = []
    for module_config in package_config.passes.values():
        if not _matches(module_config.supported_accelerators, args.device):
            continue
        if not _matches(module_config.supported_providers, args.provider):
            continue
        if not _matches(module_config.supported_precisions, args.precision):
            continue

        passes.append(
            {
                "name": module_config.module_path.rsplit(".", 1)[-1],
                "accelerators": sorted(module_config.supported_accelerators),
                "providers": sorted(module_config.supported_providers),
                "precisions": sorted(module_config.supported_precisions),
                "algorithms": sorted(module_config.supported_algorithms),
                "dataset": str(module_config.dataset),
            }
        )

    print(json.dumps({"passes": passes, "total": len(passes)}, indent=2))


def _inspect_pass(package_config: OlivePackageConfig, args: argparse.Namespace):
    module_config = package_config.get_pass_module_config(args.pass_name)
    pass_class = package_config.import_pass_module(args.pass_name)
    accelerator = AcceleratorSpec(args.device or "cpu", args.provider or "CPUExecutionProvider")
    default_config = pass_class.default_config(accelerator)

    packages = list(module_config.module_dependencies)
    for extra in module_config.extra_dependencies:
        packages.extend(package_config.extra_dependencies.get(extra, [f"olive-ai[{extra}]"]))

    parameters = {}
    for name, parameter in default_config.items():
        parameters[name] = {
            "type": _type_name(parameter.type_),
            "required": parameter.required,
            "default": _serialize(parameter.default_value),
            "search_defaults": _serialize(parameter.search_defaults),
            "description": parameter.description or "",
        }

    result = {
        "name": pass_class.__name__,
        "module_path": module_config.module_path,
        "accelerator": accelerator.to_json(),
        "supported_accelerators": sorted(module_config.supported_accelerators),
        "supported_providers": sorted(module_config.supported_providers),
        "supported_precisions": sorted(module_config.supported_precisions),
        "supported_algorithms": sorted(module_config.supported_algorithms),
        "supported_quantization_encodings": sorted(module_config.supported_quantization_encodings),
        "dataset": str(module_config.dataset),
        "packages": sorted(set(packages)),
        "parameters": parameters,
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inspect passes from the active Microsoft Olive installation.")
    parser.add_argument("pass_name", nargs="?", help="Pass type to inspect, for example OnnxConversion.")
    parser.add_argument("--list", action="store_true", help="List passes instead of inspecting one pass.")
    parser.add_argument(
        "--device",
        help="Filter list output by device; defaults to cpu when inspecting one pass.",
    )
    parser.add_argument(
        "--provider",
        help="Filter list output by provider; defaults to CPUExecutionProvider when inspecting one pass.",
    )
    parser.add_argument("--precision", help="Filter list output by supported precision.")
    args = parser.parse_args()

    if not args.list and not args.pass_name:
        parser.error("pass_name is required unless --list is used")
    if not args.list and bool(args.device) != bool(args.provider):
        parser.error("--device and --provider must be provided together when inspecting one pass")

    package_config = OlivePackageConfig.load_default_config()
    if args.list:
        _list_passes(package_config, args)
    else:
        _inspect_pass(package_config, args)


if __name__ == "__main__":
    main()
