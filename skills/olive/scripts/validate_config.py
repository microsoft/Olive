#!/usr/bin/env python
# ruff: noqa: T201

import argparse

from olive.hardware import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec, Device
from olive.hardware.constants import DEVICE_TO_EXECUTION_PROVIDERS
from olive.package_config import OlivePackageConfig
from olive.workflows.run.config import RunConfig
from olive.workflows.run.run import get_required_packages


def _accelerator_from_config(run_config: RunConfig) -> AcceleratorSpec:
    system = run_config.engine.target or run_config.engine.host
    if not system or not system.config or not system.config.accelerators:
        return DEFAULT_CPU_ACCELERATOR

    accelerator = system.config.accelerators[0]
    providers = accelerator.get_ep_strs() or []
    provider = providers[0] if providers else None
    device = accelerator.device

    if device is None and provider:
        device = next(
            (
                candidate
                for candidate, candidate_providers in DEVICE_TO_EXECUTION_PROVIDERS.items()
                if provider in candidate_providers
            ),
            Device.CPU,
        )

    return AcceleratorSpec(device or Device.CPU, provider, accelerator.memory)


def _validate_passes(
    run_config: RunConfig,
    package_config: OlivePackageConfig,
    accelerator: AcceleratorSpec,
):
    search_enabled = bool(run_config.engine.search_strategy)
    validated = []

    for name, pass_configs in run_config.passes.items():
        for index, pass_config in enumerate(pass_configs):
            pass_class = package_config.import_pass_module(pass_config.type)
            provided_config = pass_config.config or {}
            known_parameters = set(pass_class.default_config(accelerator))
            unknown_parameters = set(provided_config) - known_parameters
            if unknown_parameters:
                unknown = ", ".join(sorted(unknown_parameters))
                raise ValueError(f"Pass '{name}' has unknown parameters: {unknown}")

            pass_class.get_config_params(
                accelerator,
                provided_config,
                disable_search=not search_enabled,
            )
            display_name = name if len(pass_configs) == 1 else f"{name}[{index}]"
            validated.append((display_name, pass_class.__name__))

    return validated


def main():
    parser = argparse.ArgumentParser(description="Validate a Microsoft Olive YAML or JSON workflow.")
    parser.add_argument("config", help="Path to an Olive .yaml, .yml, or .json workflow.")
    args = parser.parse_args()

    package_config = OlivePackageConfig.load_default_config()
    run_config = RunConfig.parse_file_or_obj(args.config)
    accelerator = _accelerator_from_config(run_config)
    passes = _validate_passes(run_config, package_config, accelerator)
    required_packages = sorted(get_required_packages(package_config, run_config))

    print(f"Valid Olive workflow: {args.config}")
    print(f"Workflow ID: {run_config.workflow_id}")
    print(f"Input model type: {run_config.input_model.type}")
    print(f"Target accelerator: {accelerator}")
    print("Pass order:")
    if passes:
        for index, (name, pass_type) in enumerate(passes, start=1):
            print(f"  {index}. {name}: {pass_type}")
    else:
        print("  No explicit passes")
    print("Declared local packages:")
    if required_packages:
        for package in required_packages:
            print(f"  - {package}")
    else:
        print("  None")


if __name__ == "__main__":
    main()
