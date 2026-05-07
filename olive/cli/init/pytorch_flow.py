# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import questionary

from olive.cli.init.helpers import (
    SourceType,
    _ask,
    _ask_select,
    _device_choices,
    _precision_choices,
    build_calibration_args,
    prompt_calibration_source,
)

# PyTorch operations
OP_OPTIMIZE = "optimize"
OP_EXPORT = "export"
OP_QUANTIZE = "quantize"
OP_FINETUNE = "finetune"
OP_GRAPH_OPT = "graph_opt"

# Optimize modes
MODE_AUTO = "auto"
MODE_CUSTOM = "custom"

# Exporters
EXPORTER_MODEL_BUILDER = "model_builder"
EXPORTER_DYNAMO = "dynamo"
EXPORTER_TORCHSCRIPT = "torchscript"

# Text construction modes
TEXT_FIELD = "text_field"
TEXT_TEMPLATE = "template"
TEXT_CHAT_TEMPLATE = "chat_template"

# Precision (used in routing)
PRECISION_INT4 = "int4"

# Algorithms that require calibration data
CALIBRATION_ALGORITHMS = {"gptq", "awq", "quarot", "spinquant"}

# Algorithms that need a non-default --implementation value
# (rtn and gptq use the default "olive" implementation, so they are omitted)
ALGORITHM_TO_IMPLEMENTATION = {
    "awq": "awq",
    "quarot": "quarot",
    "spinquant": "spinquant",
}


def _build_model_args(model_config):
    """Build model-related CLI args from model config."""
    parts = []
    model_path = model_config.get("model_path")
    if model_path:
        parts.append(f"-m {model_path}")
    if model_config.get("model_script"):
        parts.append(f"--model_script {model_config['model_script']}")
    if model_config.get("script_dir"):
        parts.append(f"--script_dir {model_config['script_dir']}")
    return " ".join(parts)


def run_pytorch_flow(model_config):
    operation = _ask_select(
        "What do you want to do?",
        choices=[
            questionary.Choice("Optimize model (export to ONNX + quantize + graph optimize)", value=OP_OPTIMIZE),
            questionary.Choice("Export to ONNX only", value=OP_EXPORT),
            questionary.Choice("Quantize only (PyTorch quantization)", value=OP_QUANTIZE),
            questionary.Choice("Fine-tune model (LoRA, QLoRA)", value=OP_FINETUNE),
        ],
    )

    if operation == OP_OPTIMIZE:
        return _optimize_flow(model_config)
    elif operation == OP_EXPORT:
        return _export_flow(model_config)
    elif operation == OP_QUANTIZE:
        return _quantize_flow(model_config)
    elif operation == OP_FINETUNE:
        return _finetune_flow(model_config)
    return {}


def _optimize_flow(model_config):
    mode = _ask(
        questionary.select(
            "How would you like to configure optimization?",
            choices=[
                questionary.Choice(
                    "Auto Mode (recommended) - Automatically select best passes for your target", value=MODE_AUTO
                ),
                questionary.Choice("Custom Mode - Manually pick operations and parameters", value=MODE_CUSTOM),
            ],
        )
    )

    if mode == MODE_AUTO:
        return _optimize_auto_mode(model_config)
    else:
        return _optimize_custom_mode(model_config)


def _optimize_auto_mode(model_config):
    model_args = _build_model_args(model_config)

    provider = _ask(questionary.select("Select target device:", choices=_device_choices()))
    precision = _ask(questionary.select("Select target precision:", choices=_precision_choices()))

    cmd = f"olive optimize {model_args} --provider {provider} --precision {precision}"
    return {"command": cmd}


def _optimize_custom_mode(model_config):
    model_args = _build_model_args(model_config)

    operations = _ask(
        questionary.checkbox(
            "Select operations to perform:",
            choices=[
                questionary.Choice("Export to ONNX", value=OP_EXPORT, checked=True),
                questionary.Choice("Quantize", value=OP_QUANTIZE, checked=True),
                questionary.Choice("Graph Optimization", value=OP_GRAPH_OPT),
            ],
            instruction="(Space to toggle, Enter to confirm)",
        )
    )

    if not operations:
        print("No operations selected.")
        return {}

    export_config = None
    quant_config = None

    # Export options
    if OP_EXPORT in operations:
        export_config = _prompt_export_options()

    # Quantize options
    if OP_QUANTIZE in operations:
        quant_config = _prompt_quantize_options()

    has_export = OP_EXPORT in operations
    has_quantize = OP_QUANTIZE in operations
    has_graph_opt = OP_GRAPH_OPT in operations

    if has_export and has_quantize:
        # Combined export + quantize (±graph_opt) → use olive optimize with --exporter
        provider = _ask(questionary.select("Select target device:", choices=_device_choices()))
        precision = quant_config["precision"] if quant_config else "fp32"
        cmd = f"olive optimize {model_args} --provider {provider} --precision {precision}"
        # Pass exporter choice to olive optimize
        if export_config:
            exporter_map = {
                EXPORTER_MODEL_BUILDER: "model_builder",
                EXPORTER_DYNAMO: "dynamo_exporter",
                EXPORTER_TORCHSCRIPT: "torchscript_exporter",
            }
            exporter_arg = exporter_map.get(export_config.get("exporter"), "model_builder")
            cmd += f" --exporter {exporter_arg}"
        # Warn that olive optimize auto-selects the quantization algorithm
        algorithm = quant_config.get("algorithm") if quant_config else None
        if algorithm:
            print(
                f"\nNote: 'olive optimize' automatically selects the quantization algorithm based on"
                f" provider and precision. Your selection '{algorithm}' is used as a reference but the"
                f" actual algorithm may differ. To use '{algorithm}' exactly, select 'Quantize only'"
                f" instead.\n"
            )
    elif has_export and has_graph_opt:
        # Export + graph_opt (no quantize) → olive optimize with fp32
        provider = _ask(questionary.select("Select target device:", choices=_device_choices()))
        cmd = f"olive optimize {model_args} --provider {provider} --precision fp32"
        if export_config:
            exporter_map = {
                EXPORTER_MODEL_BUILDER: "model_builder",
                EXPORTER_DYNAMO: "dynamo_exporter",
                EXPORTER_TORCHSCRIPT: "torchscript_exporter",
            }
            exporter_arg = exporter_map.get(export_config.get("exporter"), "model_builder")
            cmd += f" --exporter {exporter_arg}"
    elif has_quantize and has_graph_opt:
        # Quantize + graph_opt (no export, ONNX input assumed) → olive optimize
        provider = _ask(questionary.select("Select target device:", choices=_device_choices()))
        precision = quant_config["precision"] if quant_config else "fp32"
        cmd = f"olive optimize {model_args} --provider {provider} --precision {precision}"
    elif has_export and export_config:
        # Export only → olive capture-onnx-graph with specific exporter options
        cmd = _build_export_command(model_args, export_config)
    elif has_quantize and quant_config:
        # Quantize only → olive quantize with specific algorithm/precision/calibration
        cmd = _build_quantize_command(model_args, quant_config)
    elif has_graph_opt:
        # Graph opt only
        provider = _ask(questionary.select("Select target device:", choices=_device_choices()))
        cmd = f"olive optimize {model_args} --provider {provider} --precision fp32"
    else:
        return {}

    return {"command": cmd}


def _build_export_command(model_args, export_config):
    """Build olive capture-onnx-graph command from export config."""
    exporter = export_config.get("exporter", EXPORTER_DYNAMO)

    if exporter == EXPORTER_MODEL_BUILDER:
        precision = export_config.get("precision", "fp16")
        cmd = f"olive capture-onnx-graph {model_args} --use_model_builder --precision {precision}"
        if precision == PRECISION_INT4 and "int4_block_size" in export_config:
            cmd += f" --int4_block_size {export_config['int4_block_size']}"
    elif exporter == EXPORTER_DYNAMO:
        torch_dtype = export_config.get("torch_dtype", "float32")
        cmd = f"olive capture-onnx-graph {model_args} --torch_dtype {torch_dtype}"
    else:
        cmd = f"olive capture-onnx-graph {model_args}"

    return cmd


def _build_quantize_command(model_args, quant_config):
    """Build olive quantize command from quantize config."""
    algorithm = quant_config.get("algorithm", "rtn")
    precision = quant_config.get("precision", "int4")

    cmd = f"olive quantize {model_args} --algorithm {algorithm} --precision {precision}"

    # Add --implementation for algorithms that need a non-default one
    impl = ALGORITHM_TO_IMPLEMENTATION.get(algorithm)
    if impl:
        cmd += f" --implementation {impl}"

    calibration = quant_config.get("calibration")
    if calibration:
        cmd += build_calibration_args(calibration)

    return cmd


def _export_flow(model_config):
    model_args = _build_model_args(model_config)

    exporter = _ask(
        questionary.select(
            "Select exporter:",
            choices=[
                questionary.Choice("Model Builder (recommended for LLMs)", value=EXPORTER_MODEL_BUILDER),
                questionary.Choice("Dynamo Exporter (general purpose)", value=EXPORTER_DYNAMO),
                questionary.Choice("TorchScript Exporter (legacy)", value=EXPORTER_TORCHSCRIPT),
            ],
        )
    )

    if exporter == EXPORTER_MODEL_BUILDER:
        precision = _ask(
            questionary.select(
                "Export precision:",
                choices=[
                    questionary.Choice("fp16", value="fp16"),
                    questionary.Choice("fp32", value="fp32"),
                    questionary.Choice("bf16", value="bf16"),
                    questionary.Choice("int4", value=PRECISION_INT4),
                ],
            )
        )

        cmd = f"olive capture-onnx-graph {model_args} --use_model_builder --precision {precision}"

        if precision == PRECISION_INT4:
            block_size = _ask(
                questionary.select(
                    "INT4 block size:",
                    choices=[
                        questionary.Choice("32 (recommended)", value="32"),
                        questionary.Choice("16", value="16"),
                        questionary.Choice("64", value="64"),
                        questionary.Choice("128", value="128"),
                        questionary.Choice("256", value="256"),
                    ],
                )
            )
            cmd += f" --int4_block_size {block_size}"

            accuracy_level = _ask(
                questionary.select(
                    "INT4 accuracy level:",
                    choices=[
                        questionary.Choice("4 (int8, recommended)", value="4"),
                        questionary.Choice("1 (fp32)", value="1"),
                        questionary.Choice("2 (fp16)", value="2"),
                        questionary.Choice("3 (bf16)", value="3"),
                    ],
                )
            )
            cmd += f" --int4_accuracy_level {accuracy_level}"

        return {"command": cmd}

    elif exporter == EXPORTER_DYNAMO:
        torch_dtype = _ask(
            questionary.select(
                "Torch dtype:",
                choices=[
                    questionary.Choice("fp32", value="float32"),
                    questionary.Choice("fp16", value="float16"),
                ],
            )
        )

        cmd = f"olive capture-onnx-graph {model_args} --torch_dtype {torch_dtype}"
        return {"command": cmd}

    else:
        # TorchScript
        cmd = f"olive capture-onnx-graph {model_args}"
        return {"command": cmd}


def _quantize_flow(model_config):
    model_args = _build_model_args(model_config)

    algorithm = _ask(
        questionary.select(
            "Select quantization algorithm:",
            choices=[
                questionary.Choice("RTN - Fast, no calibration needed", value="rtn"),
                questionary.Choice("GPTQ - High quality, requires calibration", value="gptq"),
                questionary.Choice("AWQ - Activation-aware, good for LLMs", value="awq"),
                questionary.Choice("QuaRot - For QNN/VitisAI deployment", value="quarot"),
                questionary.Choice("SpinQuant - Spin quantization", value="spinquant"),
            ],
        )
    )

    precision = _ask(
        questionary.select(
            "Precision:",
            choices=[
                questionary.Choice("int4", value="int4"),
                questionary.Choice("uint4", value="uint4"),
                questionary.Choice("int8", value="int8"),
            ],
        )
    )

    cmd = f"olive quantize {model_args} --algorithm {algorithm} --precision {precision}"

    # Add --implementation for algorithms that need a non-default one
    impl = ALGORITHM_TO_IMPLEMENTATION.get(algorithm)
    if impl:
        cmd += f" --implementation {impl}"

    # Calibration data for algorithms that need it
    if algorithm in CALIBRATION_ALGORITHMS:
        calib = prompt_calibration_source()
        if calib:
            cmd += build_calibration_args(calib)

    return {"command": cmd}


def _finetune_flow(model_config):
    model_args = _build_model_args(model_config)

    method = _ask(
        questionary.select(
            "Select fine-tuning method:",
            choices=[
                questionary.Choice("LoRA (recommended)", value="lora"),
                questionary.Choice("QLoRA (quantized, saves GPU memory)", value="qlora"),
            ],
        )
    )

    lora_r = _ask(
        questionary.select(
            "LoRA rank (r):",
            choices=[
                questionary.Choice("64 (default)", value="64"),
                questionary.Choice("4", value="4"),
                questionary.Choice("8", value="8"),
                questionary.Choice("16", value="16"),
                questionary.Choice("32", value="32"),
            ],
        )
    )

    lora_alpha = _ask(questionary.text("LoRA alpha:", default="16"))

    # Dataset
    data_source = _ask(
        questionary.select(
            "Training dataset:",
            choices=[
                questionary.Choice("HuggingFace dataset", value=SourceType.HF),
                questionary.Choice("Local file", value=SourceType.LOCAL),
            ],
        )
    )

    cmd = f"olive finetune {model_args} --method {method} --lora_r {lora_r} --lora_alpha {lora_alpha}"

    if data_source == SourceType.HF:
        data_name = _ask(
            questionary.text(
                "Dataset name:",
                default="tatsu-lab/alpaca",
            )
        )
        train_split = _ask(questionary.text("Train split:", default="train"))
        eval_split = _ask(questionary.text("Eval split (optional, press Enter to skip):", default=""))

        cmd += f" -d {data_name} --train_split {train_split}"
        if eval_split:
            cmd += f" --eval_split {eval_split}"
    else:
        data_files = _ask(
            questionary.text(
                "Path to data file(s):",
                validate=lambda x: True if x.strip() else "Please enter a file path",
            )
        )
        cmd += f" -d nouse --data_files {data_files}"

    # Text construction
    text_mode = _ask(
        questionary.select(
            "How to construct training text?",
            choices=[
                questionary.Choice("Single text field (specify column name)", value=TEXT_FIELD),
                questionary.Choice(
                    "Text template (e.g., '### Question: {prompt} \\n### Answer: {response}')", value=TEXT_TEMPLATE
                ),
                questionary.Choice("Use chat template", value=TEXT_CHAT_TEMPLATE),
            ],
        )
    )

    if text_mode == TEXT_FIELD:
        text_field = _ask(questionary.text("Text field name:", default="text"))
        cmd += f" --text_field {text_field}"
    elif text_mode == TEXT_TEMPLATE:
        template = _ask(questionary.text("Text template:"))
        cmd += f' --text_template "{template}"'
    else:
        cmd += " --use_chat_template"

    max_seq_len = _ask(questionary.text("Max sequence length:", default="1024"))
    cmd += f" --max_seq_len {max_seq_len}"

    max_samples = _ask(questionary.text("Max training samples:", default="256"))
    cmd += f" --max_samples {max_samples}"

    # Torch dtype
    torch_dtype = _ask(
        questionary.select(
            "Torch dtype for training:",
            choices=[
                questionary.Choice("bfloat16 (recommended)", value="bfloat16"),
                questionary.Choice("float16", value="float16"),
                questionary.Choice("float32", value="float32"),
            ],
        )
    )
    cmd += f" --torch_dtype {torch_dtype}"

    return {"command": cmd}


def _prompt_export_options():
    """Prompt export options for custom mode."""
    exporter = _ask(
        questionary.select(
            "Select exporter:",
            choices=[
                questionary.Choice("Model Builder (recommended for LLMs)", value=EXPORTER_MODEL_BUILDER),
                questionary.Choice("Dynamo Exporter (general purpose)", value=EXPORTER_DYNAMO),
                questionary.Choice("TorchScript Exporter (legacy)", value=EXPORTER_TORCHSCRIPT),
            ],
        )
    )

    config = {"exporter": exporter}

    if exporter == EXPORTER_MODEL_BUILDER:
        precision = _ask(
            questionary.select(
                "Export precision:",
                choices=[
                    questionary.Choice("fp16", value="fp16"),
                    questionary.Choice("fp32", value="fp32"),
                    questionary.Choice("bf16", value="bf16"),
                    questionary.Choice("int4", value=PRECISION_INT4),
                ],
            )
        )
        config["precision"] = precision

        if precision == PRECISION_INT4:
            block_size = _ask(
                questionary.select(
                    "INT4 block size:",
                    choices=[
                        questionary.Choice("32 (recommended)", value="32"),
                        questionary.Choice("16", value="16"),
                        questionary.Choice("64", value="64"),
                        questionary.Choice("128", value="128"),
                        questionary.Choice("256", value="256"),
                    ],
                )
            )
            config["int4_block_size"] = block_size

    elif exporter == EXPORTER_DYNAMO:
        torch_dtype = _ask(
            questionary.select(
                "Torch dtype:",
                choices=[
                    questionary.Choice("fp32", value="float32"),
                    questionary.Choice("fp16", value="float16"),
                ],
            )
        )
        config["torch_dtype"] = torch_dtype

    return config


def _prompt_quantize_options():
    """Prompt quantization options for custom mode."""
    algorithm = _ask(
        questionary.select(
            "Select quantization algorithm:",
            choices=[
                questionary.Choice("RTN - Round-to-Nearest, fast, no calibration needed", value="rtn"),
                questionary.Choice("GPTQ - High quality, requires calibration data", value="gptq"),
                questionary.Choice("AWQ - Activation-aware, good for LLMs", value="awq"),
                questionary.Choice("QuaRot - Rotation-based, for QNN/VitisAI deployment", value="quarot"),
                questionary.Choice("SpinQuant - Spin quantization", value="spinquant"),
            ],
        )
    )

    precision = _ask(
        questionary.select(
            "Quantization precision:",
            choices=[
                questionary.Choice("int4", value="int4"),
                questionary.Choice("uint4", value="uint4"),
                questionary.Choice("int8", value="int8"),
            ],
        )
    )

    config = {"algorithm": algorithm, "precision": precision}

    if algorithm in CALIBRATION_ALGORITHMS:
        calib = prompt_calibration_source()
        if calib:
            config["calibration"] = calib

    return config
