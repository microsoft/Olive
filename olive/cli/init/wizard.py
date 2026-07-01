# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import subprocess
import sys
from pathlib import Path

import questionary

from olive.cli.init.helpers import (
    DiffuserVariant,
    GoBackError,
    SourceType,
    _ask,
    _ask_select,
)
from olive.common.utils import StrEnumBase


class ModelType(StrEnumBase):
    """Model types."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    DIFFUSERS = "diffusers"


class OutputAction(StrEnumBase):
    """Output actions."""

    COMMAND = "command"
    CONFIG = "config"
    RUN = "run"


class InitWizard:
    def __init__(self, default_output_path: str = "./olive-output"):
        self.default_output_path = default_output_path

    def start(self):
        print("\nWelcome to Olive Init! This wizard will help you optimize your model.\n")

        try:
            step = 0
            model_type = None
            model_config = None
            result = None

            while step < 4:
                try:
                    if step == 0:
                        model_type = self._prompt_model_type()
                    elif step == 1:
                        model_config = self._prompt_model_source(model_type)
                    elif step == 2:
                        result = self._run_model_flow(model_type, model_config)
                    elif step == 3:
                        self._prompt_output(result)
                    step += 1
                except GoBackError:
                    if step > 0:
                        step -= 1

        except KeyboardInterrupt:
            sys.exit(0)

    def _prompt_model_type(self):
        return _ask_select(
            "What type of model do you want to optimize?",
            choices=[
                questionary.Choice("PyTorch (HuggingFace or local)", value=ModelType.PYTORCH),
                questionary.Choice("ONNX", value=ModelType.ONNX),
                questionary.Choice("Diffusers (Stable Diffusion, SDXL, Flux, etc.)", value=ModelType.DIFFUSERS),
            ],
            allow_back=False,
        )

    def _prompt_model_source(self, model_type):
        if model_type == ModelType.PYTORCH:
            return self._prompt_pytorch_source()
        elif model_type == ModelType.ONNX:
            return self._prompt_onnx_source()
        elif model_type == ModelType.DIFFUSERS:
            return self._prompt_diffusers_source()
        return {}

    def _prompt_pytorch_source(self):
        source_type = _ask_select(
            "How would you like to specify your model?",
            choices=[
                questionary.Choice("HuggingFace model name (e.g., meta-llama/Llama-3.1-8B)", value=SourceType.HF),
                questionary.Choice("Local directory path", value=SourceType.LOCAL),
                questionary.Choice("AzureML registry path", value=SourceType.AZUREML),
                questionary.Choice("PyTorch model with custom script", value=SourceType.SCRIPT),
            ],
        )

        config = {"source_type": source_type}

        if source_type == SourceType.SCRIPT:
            config["model_script"] = _ask(
                questionary.path(
                    "Path to model script (.py):",
                )
            )
            script_dir = _ask(
                questionary.text(
                    "Script directory (optional, press Enter to skip):",
                    default="",
                )
            )
            if script_dir:
                config["script_dir"] = script_dir
            model_path = _ask(
                questionary.text(
                    "Model name or path (optional, press Enter to skip):",
                    default="",
                )
            )
            if model_path:
                config["model_path"] = model_path
        else:
            if source_type == SourceType.HF:
                placeholder = "e.g., meta-llama/Llama-3.1-8B"
            elif source_type == SourceType.AZUREML:
                placeholder = "e.g., azureml://registries/<registry>/models/<model>/versions/<version>"
            else:
                placeholder = "e.g., ./my-model/"
            config["model_path"] = _ask(
                questionary.text(
                    "Model name or path:",
                    validate=lambda x: True if x.strip() else "Please enter a model name or path",
                    instruction=placeholder,
                )
            )

        return config

    def _prompt_onnx_source(self):
        model_path = _ask(
            questionary.text(
                "Enter ONNX model path (file or directory):",
                validate=lambda x: True if x.strip() else "Please enter a model path",
            )
        )
        return {"source_type": SourceType.LOCAL, "model_path": model_path}

    def _prompt_diffusers_source(self):
        variant = _ask_select(
            "Select diffuser model variant:",
            choices=[
                questionary.Choice("Auto-detect", value=DiffuserVariant.AUTO),
                questionary.Choice("Stable Diffusion (SD 1.x/2.x)", value="sd"),
                questionary.Choice("Stable Diffusion XL (SDXL)", value="sdxl"),
                questionary.Choice("Stable Diffusion 3 (SD3)", value="sd3"),
                questionary.Choice("Flux", value=DiffuserVariant.FLUX),
                questionary.Choice("Sana", value="sana"),
            ],
        )

        model_path = _ask(
            questionary.text(
                "Enter model name or path:",
                validate=lambda x: True if x.strip() else "Please enter a model name or path",
                instruction="e.g., stabilityai/stable-diffusion-xl-base-1.0",
            )
        )

        return {"source_type": SourceType.HF, "model_path": model_path, "variant": variant}

    def _run_model_flow(self, model_type, model_config):
        if model_type == ModelType.PYTORCH:
            from olive.cli.init.pytorch_flow import run_pytorch_flow

            return run_pytorch_flow(model_config)
        elif model_type == ModelType.ONNX:
            from olive.cli.init.onnx_flow import run_onnx_flow

            return run_onnx_flow(model_config)
        elif model_type == ModelType.DIFFUSERS:
            from olive.cli.init.diffusers_flow import run_diffusers_flow

            return run_diffusers_flow(model_config)
        return {}

    def _prompt_output(self, result):
        command_str = result.get("command")

        if not command_str:
            print("No command generated.")
            raise GoBackError

        output_dir = _ask(
            questionary.text(
                "Output directory:",
                default=self.default_output_path,
            )
        )

        # Append output dir to command if not already present
        if " -o " not in command_str and " --output_path " not in command_str:
            command_str += f" -o {output_dir}"

        action = _ask_select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Generate CLI command (copy and run later)", value=OutputAction.COMMAND),
                questionary.Choice("Generate configuration file (JSON, for olive run)", value=OutputAction.CONFIG),
                questionary.Choice("Run optimization now", value=OutputAction.RUN),
            ],
        )

        if action == OutputAction.COMMAND:
            print(f"\nGenerated command:\n\n  {command_str}\n")
            run_now = _ask(questionary.confirm("Run this command now?", default=False))
            if run_now:
                print(f"\nRunning: {command_str}\n")
                subprocess.run(command_str, shell=True, check=False)

        elif action == OutputAction.CONFIG:
            config_cmd = command_str + " --save_config_file --dry_run"
            print("\nGenerating configuration file...\n")
            subprocess.run(config_cmd, shell=True, check=False)
            config_path = Path(output_dir) / "config.json"
            if config_path.exists():
                print(f"\nYou can run it later with:\n  olive run --config {config_path}\n")

        elif action == OutputAction.RUN:
            print(f"\nRunning: {command_str}\n")
            subprocess.run(command_str, shell=True, check=False)
