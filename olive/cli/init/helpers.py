# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared constants, helpers and UI prompts used by the init wizard and flow modules."""

import sys

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

# Source types (shared across flows)
SOURCE_HF = "hf"
SOURCE_LOCAL = "local"
SOURCE_AZUREML = "azureml"
SOURCE_SCRIPT = "script"
SOURCE_DEFAULT = "default"

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
        return {
            "source": SOURCE_HF,
            "data_name": data_name,
            "subset": subset,
            "split": split,
            "num_samples": num_samples,
        }
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
