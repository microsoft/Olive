# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import questionary

from olive.cli.init.wizard import _ask, _ask_select

# Device choices -> provider mapping
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

# Algorithms that require calibration data
CALIBRATION_ALGORITHMS = {"gptq", "awq", "quarot", "spinquant"}


def run_pytorch_flow(model_config):
    operation = _ask_select(
        "What do you want to do?",
        choices=[
            questionary.Choice("Optimize model (export to ONNX + quantize + graph optimize)", value="optimize"),
            questionary.Choice("Export to ONNX only", value="export"),
            questionary.Choice("Quantize only (PyTorch quantization)", value="quantize"),
            questionary.Choice("Fine-tune model (LoRA, QLoRA)", value="finetune"),
        ],
    )

    if operation == "optimize":
        return _optimize_flow(model_config)
    elif operation == "export":
        return _export_flow(model_config)
    elif operation == "quantize":
        return _quantize_flow(model_config)
    elif operation == "finetune":
        return _finetune_flow(model_config)
    return {}


def _optimize_flow(model_config):
    mode = _ask(questionary.select(
        "How would you like to configure optimization?",
        choices=[
            questionary.Choice("Auto Mode (recommended) - Automatically select best passes for your target", value="auto"),
            questionary.Choice("Custom Mode - Manually pick operations and parameters", value="custom"),
        ],
    ))

    if mode == "auto":
        return _optimize_auto_mode(model_config)
    else:
        return _optimize_custom_mode(model_config)


def _optimize_auto_mode(model_config):
    model_path = model_config.get("model_path", "")

    provider = _ask(questionary.select("Select target device:", choices=DEVICE_CHOICES))
    precision = _ask(questionary.select("Select target precision:", choices=PRECISION_CHOICES))

    cmd = f"olive optimize -m {model_path} --provider {provider} --precision {precision}"
    return {"command": cmd}


def _optimize_custom_mode(model_config):
    model_path = model_config.get("model_path", "")

    operations = _ask(questionary.checkbox(
        "Select operations to perform:",
        choices=[
            questionary.Choice("Export to ONNX", value="export", checked=True),
            questionary.Choice("Quantize", value="quantize", checked=True),
            questionary.Choice("Graph Optimization", value="graph_opt"),
        ],
    ))

    cmd_parts = []

    # Export options
    if "export" in operations:
        exporter_config = _prompt_export_options()
        cmd_parts.append(exporter_config)

    # Quantize options
    if "quantize" in operations:
        quant_config = _prompt_quantize_options()
        cmd_parts.append(quant_config)

    # Graph optimization options
    if "graph_opt" in operations:
        _ask(questionary.checkbox(
            "Select optimizations:",
            choices=[
                questionary.Choice("Peephole optimization", value="peephole", checked=True),
                questionary.Choice("Transformer optimization", value="transformer", checked=True),
            ],
        ))

    # Ask for target device/provider
    provider = _ask(questionary.select("Select target device:", choices=DEVICE_CHOICES))
    precision = _ask(questionary.select("Select target precision:", choices=PRECISION_CHOICES))

    # Build the combined command - for custom mode with export+quantize, use olive optimize
    cmd = f"olive optimize -m {model_path} --provider {provider} --precision {precision}"
    return {"command": cmd}


def _export_flow(model_config):
    model_path = model_config.get("model_path", "")

    exporter = _ask(questionary.select(
        "Select exporter:",
        choices=[
            questionary.Choice("Model Builder (recommended for LLMs)", value="model_builder"),
            questionary.Choice("Dynamo Exporter (general purpose)", value="dynamo"),
            questionary.Choice("TorchScript Exporter (legacy)", value="torchscript"),
        ],
    ))

    if exporter == "model_builder":
        precision = _ask(questionary.select(
            "Export precision:",
            choices=[
                questionary.Choice("fp16", value="fp16"),
                questionary.Choice("fp32", value="fp32"),
                questionary.Choice("bf16", value="bf16"),
                questionary.Choice("int4", value="int4"),
            ],
        ))

        cmd = f"olive capture-onnx-graph -m {model_path} --use_model_builder --precision {precision}"

        if precision == "int4":
            block_size = _ask(questionary.select(
                "INT4 block size:",
                choices=[
                    questionary.Choice("32 (recommended)", value="32"),
                    questionary.Choice("16", value="16"),
                    questionary.Choice("64", value="64"),
                    questionary.Choice("128", value="128"),
                    questionary.Choice("256", value="256"),
                ],
            ))
            cmd += f" --int4_block_size {block_size}"

            accuracy_level = _ask(questionary.select(
                "INT4 accuracy level:",
                choices=[
                    questionary.Choice("4 (int8, recommended)", value="4"),
                    questionary.Choice("1 (fp32)", value="1"),
                    questionary.Choice("2 (fp16)", value="2"),
                    questionary.Choice("3 (bf16)", value="3"),
                ],
            ))
            cmd += f" --int4_accuracy_level {accuracy_level}"

        return {"command": cmd}

    elif exporter == "dynamo":
        torch_dtype = _ask(questionary.select(
            "Torch dtype:",
            choices=[
                questionary.Choice("fp32", value="float32"),
                questionary.Choice("fp16", value="float16"),
            ],
        ))

        cmd = f"olive capture-onnx-graph -m {model_path} --torch_dtype {torch_dtype}"
        return {"command": cmd}

    else:
        # TorchScript
        cmd = f"olive capture-onnx-graph -m {model_path}"
        return {"command": cmd}


def _quantize_flow(model_config):
    model_path = model_config.get("model_path", "")

    algorithm = _ask(questionary.select(
        "Select quantization algorithm:",
        choices=[
            questionary.Choice("RTN - Fast, no calibration needed", value="rtn"),
            questionary.Choice("GPTQ - High quality, requires calibration", value="gptq"),
            questionary.Choice("AWQ - Activation-aware, good for LLMs", value="awq"),
            questionary.Choice("QuaRot - For QNN/VitisAI deployment", value="quarot"),
            questionary.Choice("SpinQuant - Spin quantization", value="spinquant"),
        ],
    ))

    precision = _ask(questionary.select(
        "Precision:",
        choices=[
            questionary.Choice("int4", value="int4"),
            questionary.Choice("uint4", value="uint4"),
            questionary.Choice("int8", value="int8"),
        ],
    ))

    cmd = f"olive quantize -m {model_path} --algorithm {algorithm} --precision {precision}"

    # Calibration data for algorithms that need it
    if algorithm in CALIBRATION_ALGORITHMS:
        calib = _prompt_calibration_data()
        if calib:
            cmd += calib

    return {"command": cmd}


def _finetune_flow(model_config):
    model_path = model_config.get("model_path", "")

    method = _ask(questionary.select(
        "Select fine-tuning method:",
        choices=[
            questionary.Choice("LoRA (recommended)", value="lora"),
            questionary.Choice("QLoRA (quantized, saves GPU memory)", value="qlora"),
        ],
    ))

    lora_r = _ask(questionary.select(
        "LoRA rank (r):",
        choices=[
            questionary.Choice("64 (default)", value="64"),
            questionary.Choice("4", value="4"),
            questionary.Choice("8", value="8"),
            questionary.Choice("16", value="16"),
            questionary.Choice("32", value="32"),
        ],
    ))

    lora_alpha = _ask(questionary.text("LoRA alpha:", default="16"))

    # Dataset
    data_source = _ask(questionary.select(
        "Training dataset:",
        choices=[
            questionary.Choice("HuggingFace dataset", value="hf"),
            questionary.Choice("Local file", value="local"),
        ],
    ))

    cmd = f"olive finetune -m {model_path} --method {method} --lora_r {lora_r} --lora_alpha {lora_alpha}"

    if data_source == "hf":
        data_name = _ask(questionary.text(
            "Dataset name:",
            default="tatsu-lab/alpaca",
        ))
        train_split = _ask(questionary.text("Train split:", default="train"))
        eval_split = _ask(questionary.text("Eval split (optional, press Enter to skip):", default=""))

        cmd += f" -d {data_name} --train_split {train_split}"
        if eval_split:
            cmd += f" --eval_split {eval_split}"
    else:
        data_files = _ask(questionary.text(
            "Path to data file(s):",
            validate=lambda x: True if x.strip() else "Please enter a file path",
        ))
        cmd += f" -d nouse --data_files {data_files}"

    # Text construction
    text_mode = _ask(questionary.select(
        "How to construct training text?",
        choices=[
            questionary.Choice("Single text field (specify column name)", value="text_field"),
            questionary.Choice("Text template (e.g., '### Question: {prompt} \\n### Answer: {response}')", value="template"),
            questionary.Choice("Use chat template", value="chat_template"),
        ],
    ))

    if text_mode == "text_field":
        text_field = _ask(questionary.text("Text field name:", default="text"))
        cmd += f" --text_field {text_field}"
    elif text_mode == "template":
        template = _ask(questionary.text("Text template:"))
        cmd += f' --text_template "{template}"'
    else:
        cmd += " --use_chat_template"

    max_seq_len = _ask(questionary.text("Max sequence length:", default="1024"))
    cmd += f" --max_seq_len {max_seq_len}"

    max_samples = _ask(questionary.text("Max training samples:", default="256"))
    cmd += f" --max_samples {max_samples}"

    # Torch dtype
    torch_dtype = _ask(questionary.select(
        "Torch dtype for training:",
        choices=[
            questionary.Choice("bfloat16 (recommended)", value="bfloat16"),
            questionary.Choice("float16", value="float16"),
            questionary.Choice("float32", value="float32"),
        ],
    ))
    cmd += f" --torch_dtype {torch_dtype}"

    return {"command": cmd}


def _prompt_export_options():
    """Prompt export options for custom mode."""
    exporter = _ask(questionary.select(
        "Select exporter:",
        choices=[
            questionary.Choice("Model Builder (recommended for LLMs)", value="model_builder"),
            questionary.Choice("Dynamo Exporter (general purpose)", value="dynamo"),
            questionary.Choice("TorchScript Exporter (legacy)", value="torchscript"),
        ],
    ))

    config = {"exporter": exporter}

    if exporter == "model_builder":
        precision = _ask(questionary.select(
            "Export precision:",
            choices=[
                questionary.Choice("fp16", value="fp16"),
                questionary.Choice("fp32", value="fp32"),
                questionary.Choice("bf16", value="bf16"),
                questionary.Choice("int4", value="int4"),
            ],
        ))
        config["precision"] = precision

        if precision == "int4":
            block_size = _ask(questionary.select(
                "INT4 block size:",
                choices=[
                    questionary.Choice("32 (recommended)", value="32"),
                    questionary.Choice("16", value="16"),
                    questionary.Choice("64", value="64"),
                    questionary.Choice("128", value="128"),
                    questionary.Choice("256", value="256"),
                ],
            ))
            config["int4_block_size"] = block_size

    elif exporter == "dynamo":
        torch_dtype = _ask(questionary.select(
            "Torch dtype:",
            choices=[
                questionary.Choice("fp32", value="float32"),
                questionary.Choice("fp16", value="float16"),
            ],
        ))
        config["torch_dtype"] = torch_dtype

    return config


def _prompt_quantize_options():
    """Prompt quantization options for custom mode."""
    algorithm = _ask(questionary.select(
        "Select quantization algorithm:",
        choices=[
            questionary.Choice("RTN - Round-to-Nearest, fast, no calibration needed", value="rtn"),
            questionary.Choice("GPTQ - High quality, requires calibration data", value="gptq"),
            questionary.Choice("AWQ - Activation-aware, good for LLMs", value="awq"),
            questionary.Choice("QuaRot - Rotation-based, for QNN/VitisAI deployment", value="quarot"),
            questionary.Choice("SpinQuant - Spin quantization", value="spinquant"),
        ],
    ))

    precision = _ask(questionary.select(
        "Quantization precision:",
        choices=[
            questionary.Choice("int4", value="int4"),
            questionary.Choice("uint4", value="uint4"),
            questionary.Choice("int8", value="int8"),
        ],
    ))

    config = {"algorithm": algorithm, "precision": precision}

    if algorithm in CALIBRATION_ALGORITHMS:
        calib = _prompt_calibration_source()
        if calib:
            config["calibration"] = calib

    return config


def _prompt_calibration_data():
    """Prompt for calibration data and return CLI args string."""
    source = _ask(questionary.select(
        "Calibration data source:",
        choices=[
            questionary.Choice("Use default (wikitext-2)", value="default"),
            questionary.Choice("HuggingFace dataset", value="hf"),
            questionary.Choice("Local file", value="local"),
        ],
    ))

    if source == "default":
        return ""
    elif source == "hf":
        data_name = _ask(questionary.text("Dataset name:", default="Salesforce/wikitext"))
        subset = _ask(questionary.text("Subset (optional):", default="wikitext-2-raw-v1"))
        split = _ask(questionary.text("Split:", default="train"))
        num_samples = _ask(questionary.text("Number of samples:", default="128"))

        result = f" -d {data_name}"
        if subset:
            result += f" --train_subset {subset}"
        result += f" --train_split {split} --max_samples {num_samples}"
        return result
    else:
        data_files = _ask(questionary.text("Data file path:"))
        return f" --data_files {data_files}"


def _prompt_calibration_source():
    """Prompt for calibration data source (returns dict for config use)."""
    source = _ask(questionary.select(
        "Calibration data source:",
        choices=[
            questionary.Choice("Use default (wikitext-2)", value="default"),
            questionary.Choice("HuggingFace dataset", value="hf"),
            questionary.Choice("Local file", value="local"),
        ],
    ))

    if source == "default":
        return None
    elif source == "hf":
        data_name = _ask(questionary.text("Dataset name:", default="Salesforce/wikitext"))
        subset = _ask(questionary.text("Subset (optional):", default="wikitext-2-raw-v1"))
        split = _ask(questionary.text("Split:", default="train"))
        num_samples = _ask(questionary.text("Number of samples:", default="128"))
        return {"source": "hf", "data_name": data_name, "subset": subset, "split": split, "num_samples": num_samples}
    else:
        data_files = _ask(questionary.text("Data file path:"))
        return {"source": "local", "data_files": data_files}
