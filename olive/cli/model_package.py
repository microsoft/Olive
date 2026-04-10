# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from olive.cli.base import (
    BaseOliveCLICommand,
    add_logging_options,
    add_save_config_file_options,
    add_telemetry_options,
)
from olive.telemetry import action

logger = logging.getLogger(__name__)


class ModelPackageCommand(BaseOliveCLICommand):
    """Merge multiple model outputs into a model package via the ModelPackage pass."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "generate-model-package",
            help="Merge multiple model outputs into a model package with manifest",
        )

        sub_parser.add_argument(
            "-s",
            "--source",
            type=str,
            action="append",
            required=True,
            help="Source Olive output directory. Can be specified multiple times.",
        )

        sub_parser.add_argument(
            "-o",
            "--output_path",
            type=str,
            required=True,
            help="Output directory for the merged model package.",
        )

        sub_parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="Model name for the manifest. If not set, derived from the output directory name.",
        )

        sub_parser.add_argument(
            "--model_version",
            type=str,
            default="1.0",
            help="Model version string for the manifest. Default: 1.0",
        )

        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=ModelPackageCommand)

    def _get_run_config(self, tempdir: str) -> dict[str, Any]:
        sources = self._parse_sources()

        target_models = []
        target_names = []
        for target_name, source_path in sources:
            model_config = self._read_model_config(source_path)
            target_models.append(model_config)
            target_names.append(target_name)

        ep, device = self._extract_accelerator_info(target_models)

        return {
            "input_model": {
                "type": "ModelPackageModel",
                "target_models": target_models,
                "target_names": target_names,
                "model_path": tempdir,
            },
            "systems": {
                "local_system": {
                    "type": "LocalSystem",
                    "accelerators": [{"device": device, "execution_providers": [ep]}],
                }
            },
            "passes": {
                "pkg": {
                    "type": "ModelPackage",
                    "model_name": self.args.model_name,
                    "model_version": self.args.model_version,
                }
            },
            "output_dir": self.args.output_path,
            "host": "local_system",
            "target": "local_system",
            "log_severity_level": self.args.log_level,
            "no_artifacts": True,
        }

    @action
    def run(self):
        return self._run_workflow()

    def _parse_sources(self) -> list[tuple[str, Path]]:
        sources = []
        for source in self.args.source:
            path = Path(source)
            if not path.is_dir():
                raise ValueError(f"Source path does not exist or is not a directory: {path}")

            if not (path / "model_config.json").exists():
                raise ValueError(
                    f"No model_config.json found in {path}. "
                    "Source must be an Olive output directory with model_config.json."
                )

            sources.append((path.name, path))

        if len(sources) < 2:
            raise ValueError("At least two --source directories are required to merge.")

        return sources

    @staticmethod
    def _read_model_config(source_path: Path) -> dict:
        config_path = source_path / "model_config.json"
        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def _extract_accelerator_info(target_models: list[dict]) -> tuple[str, str]:
        for model_config in target_models:
            attrs = model_config.get("config", {}).get("model_attributes") or {}
            ep = attrs.get("ep", "CPUExecutionProvider")
            device = attrs.get("device", "cpu")
            return ep, device.lower()
        return "CPUExecutionProvider", "cpu"
