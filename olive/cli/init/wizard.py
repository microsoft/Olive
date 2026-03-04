# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import subprocess
import sys
from pathlib import Path

import questionary

# Common choices shared across flows — aligned with olive optimize --provider / --precision
DEVICE_CHOICES = [
    questionary.Choice("CPU", value="CPUExecutionProvider"),
    questionary.Choice("GPU (NVIDIA CUDA)", value="CUDAExecutionProvider"),
    questionary.Choice("GPU (NvTensorRTRTX)", value="NvTensorRTRTXExecutionProvider"),
    questionary.Choice("NPU (Qualcomm QNN)", value="QNNExecutionProvider"),
    questionary.Choice("NPU (Intel OpenVINO)", value="OpenVINOExecutionProvider"),
    questionary.Choice("NPU (AMD Vitis AI)", value="VitisAIExecutionProvider"),
    questionary.Choice("WebGPU", value="WebGpuExecutionProvider"),
]

PRECISION_CHOICES = [
    questionary.Choice("INT4 (smallest size, best for LLMs)", value="int4"),
    questionary.Choice("INT8 (balanced)", value="int8"),
    questionary.Choice("FP16 (half precision)", value="fp16"),
    questionary.Choice("FP32 (full precision)", value="fp32"),
]

# Model types
MODEL_PYTORCH = "pytorch"
MODEL_ONNX = "onnx"
MODEL_DIFFUSERS = "diffusers"

# Source types (shared across flows)
SOURCE_HF = "hf"
SOURCE_LOCAL = "local"
SOURCE_AZUREML = "azureml"
SOURCE_SCRIPT = "script"
SOURCE_DEFAULT = "default"

# Output actions
ACTION_COMMAND = "command"
ACTION_CONFIG = "config"
ACTION_RUN = "run"

# Diffuser variants (only those used in routing)
VARIANT_AUTO = "auto"
VARIANT_FLUX = "flux"


class GoBackError(Exception):
    """Raised when user wants to go back to the previous wizard step."""


_BACK = "__back__"


def _ask(question):
    """Ask a questionary question and handle Ctrl+C (returns None)."""
    result = question.ask()
    if result is None:
        sys.exit(0)
    return result


def _ask_select(message, choices, allow_back=True):
    """Ask a select question with optional Back choice."""
    all_choices = list(choices)
    if allow_back:
        all_choices.append(questionary.Choice("\u2190 Back", value=_BACK))
    result = _ask(questionary.select(message, choices=all_choices))
    if result == _BACK:
        raise GoBackError
    return result


def prompt_calibration_source():
    """Prompt for calibration data source. Returns dict or None (for default)."""
    source = _ask(
        questionary.select(
            "Calibration data source:",
            choices=[
                questionary.Choice("Use default (wikitext-2)", value=SOURCE_DEFAULT),
                questionary.Choice("HuggingFace dataset", value=SOURCE_HF),
                questionary.Choice("Local file", value=SOURCE_LOCAL),
            ],
        )
    )

    if source == SOURCE_DEFAULT:
        return None
    elif source == SOURCE_HF:
        data_name = _ask(questionary.text("Dataset name:", default="Salesforce/wikitext"))
        subset = _ask(questionary.text("Subset (optional):", default="wikitext-2-raw-v1"))
        split = _ask(questionary.text("Split:", default="train"))
        num_samples = _ask(questionary.text("Number of samples:", default="128"))
        return {"source": SOURCE_HF, "data_name": data_name, "subset": subset, "split": split, "num_samples": num_samples}
    else:
        data_files = _ask(questionary.text("Data file path:"))
        return {"source": SOURCE_LOCAL, "data_files": data_files}


def build_calibration_args(calibration):
    """Build CLI args string from calibration config dict."""
    if calibration["source"] == SOURCE_HF:
        result = f" -d {calibration['data_name']}"
        if calibration.get("subset"):
            result += f" --subset {calibration['subset']}"
        result += f" --split {calibration['split']} --max_samples {calibration['num_samples']}"
        return result
    elif calibration["source"] == SOURCE_LOCAL:
        return f" --data_files {calibration['data_files']}"
    return ""


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
                questionary.Choice("PyTorch (HuggingFace or local)", value=MODEL_PYTORCH),
                questionary.Choice("ONNX", value=MODEL_ONNX),
                questionary.Choice("Diffuser (Stable Diffusion, SDXL, Flux, etc.)", value=MODEL_DIFFUSERS),
            ],
            allow_back=False,
        )

    def _prompt_model_source(self, model_type):
        if model_type == MODEL_PYTORCH:
            return self._prompt_pytorch_source()
        elif model_type == MODEL_ONNX:
            return self._prompt_onnx_source()
        elif model_type == MODEL_DIFFUSERS:
            return self._prompt_diffusers_source()
        return {}

    def _prompt_pytorch_source(self):
        source_type = _ask_select(
            "How would you like to specify your model?",
            choices=[
                questionary.Choice("HuggingFace model name (e.g., meta-llama/Llama-3.1-8B)", value=SOURCE_HF),
                questionary.Choice("Local directory path", value=SOURCE_LOCAL),
                questionary.Choice("AzureML registry path", value=SOURCE_AZUREML),
                questionary.Choice("PyTorch model with custom script", value=SOURCE_SCRIPT),
            ],
        )

        config = {"source_type": source_type}

        if source_type == SOURCE_SCRIPT:
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
            if source_type == SOURCE_HF:
                placeholder = "e.g., meta-llama/Llama-3.1-8B"
            elif source_type == SOURCE_AZUREML:
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
        return {"source_type": SOURCE_LOCAL, "model_path": model_path}

    def _prompt_diffusers_source(self):
        variant = _ask_select(
            "Select diffuser model variant:",
            choices=[
                questionary.Choice("Auto-detect", value=VARIANT_AUTO),
                questionary.Choice("Stable Diffusion (SD 1.x/2.x)", value="sd"),
                questionary.Choice("Stable Diffusion XL (SDXL)", value="sdxl"),
                questionary.Choice("Stable Diffusion 3 (SD3)", value="sd3"),
                questionary.Choice("Flux", value=VARIANT_FLUX),
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

        return {"source_type": SOURCE_HF, "model_path": model_path, "variant": variant}

    def _run_model_flow(self, model_type, model_config):
        if model_type == MODEL_PYTORCH:
            from olive.cli.init.pytorch_flow import run_pytorch_flow

            return run_pytorch_flow(model_config)
        elif model_type == MODEL_ONNX:
            from olive.cli.init.onnx_flow import run_onnx_flow

            return run_onnx_flow(model_config)
        elif model_type == MODEL_DIFFUSERS:
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
                questionary.Choice("Generate CLI command (copy and run later)", value=ACTION_COMMAND),
                questionary.Choice("Generate configuration file (JSON, for olive run)", value=ACTION_CONFIG),
                questionary.Choice("Run optimization now", value=ACTION_RUN),
            ],
        )

        if action == ACTION_COMMAND:
            print(f"\nGenerated command:\n\n  {command_str}\n")
            run_now = _ask(questionary.confirm("Run this command now?", default=False))
            if run_now:
                print(f"\nRunning: {command_str}\n")
                subprocess.run(command_str, shell=True, check=False)

        elif action == ACTION_CONFIG:
            config_cmd = command_str + " --save_config_file --dry_run"
            print("\nGenerating configuration file...\n")
            subprocess.run(config_cmd, shell=True, check=False)
            config_path = Path(output_dir) / "config.json"
            if config_path.exists():
                print(f"\nYou can run it later with:\n  olive run --config {config_path}\n")

        elif action == ACTION_RUN:
            print(f"\nRunning: {command_str}\n")
            subprocess.run(command_str, shell=True, check=False)
