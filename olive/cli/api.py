# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
from argparse import ArgumentParser, Namespace
from typing import Any

from olive.cli.capture_onnx import CaptureOnnxGraphCommand
from olive.cli.convert_adapters import ConvertAdaptersCommand
from olive.cli.extract_adapters import ExtractAdaptersCommand
from olive.cli.finetune import FineTuneCommand
from olive.cli.generate_adapter import GenerateAdapterCommand
from olive.cli.generate_cost_model import GenerateCostModelCommand
from olive.cli.optimize import OptimizeCommand
from olive.cli.quantize import QuantizeCommand
from olive.cli.session_params_tuning import SessionParamsTuningCommand
from olive.engine.output import WorkflowOutput

# pylint: disable=W0212


def _get_command_args_schema(command_class) -> dict[str, Any]:
    """Extract all supported arguments and their defaults from a CLI command class.

    Args:
        command_class: The CLI command class (e.g., CaptureOnnxGraphCommand)

    Returns:
        Dict mapping argument names to their default values and metadata

    """
    # Create a temporary parser to extract argument schema
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    # Register the command's subparser to get all its arguments
    command_class.register_subcommand(sub_parsers)

    # Get the subparser that was created from the 'choices' attribute
    if not sub_parsers.choices:
        return {}

    # Assuming only one subparser is registered for the command_class
    subparser = next(iter(sub_parsers.choices.values()))

    # Extract argument information
    args_schema = {}
    for action in subparser._actions:
        if action.dest == "help":
            continue

        # Get the argument name (remove leading dashes)
        arg_name = action.dest

        # Get default value
        default_value = action.default

        # Handle special cases for argument types
        if hasattr(action, "type") and action.type:
            arg_type = action.type
        elif hasattr(action, "choices") and action.choices:
            arg_type = type(next(iter(action.choices))) if action.choices else str
        else:
            arg_type = str

        # Handle boolean flags
        if action.__class__.__name__ in ["_StoreTrueAction", "_StoreFalseAction"]:
            arg_type = bool
            if default_value is None:
                default_value = action.__class__.__name__ != "_StoreTrueAction"

        args_schema[arg_name] = {
            "default": default_value,
            "type": arg_type,
            "required": action.required if hasattr(action, "required") else False,
        }

    return args_schema


def _create_unified_args(command_class, provided_kwargs: dict) -> Namespace:
    """Create a Namespace with all required arguments for a command.

    Args:
        command_class: The CLI command class
        provided_kwargs: User-provided keyword arguments

    Returns:
        Namespace object with all required arguments set

    """
    # Get the command's argument schema
    args_schema = _get_command_args_schema(command_class)

    # Create the args namespace
    args_dict = {}

    # Set all arguments with their defaults or provided values
    for arg_name, arg_info in args_schema.items():
        if arg_name in provided_kwargs:
            # Use provided value
            args_dict[arg_name] = provided_kwargs[arg_name]
        else:
            # Use default value
            args_dict[arg_name] = arg_info["default"]

    return Namespace(**args_dict)


def _run_unified_command(command_class, **kwargs) -> Any:
    """Unified command runner that works with any CLI command class.

    Args:
        command_class: The CLI command class to run
        **kwargs: Keyword arguments to pass to the command

    Returns:
        The output of the command's run() method, which can be WorkflowOutput or None.

    """
    # Separate known from unknown args for commands like finetune
    args_schema = _get_command_args_schema(command_class)
    known_kwargs = {k: v for k, v in kwargs.items() if k in args_schema}

    args = _create_unified_args(command_class, known_kwargs)

    # Check if command handles unknown_args (like FineTuneCommand)
    constructor_params = inspect.signature(command_class.__init__).parameters
    if "unknown_args" in constructor_params:
        unknown_args_list = []
        for k, v in kwargs.items():
            if k not in known_kwargs:
                unknown_args_list.extend([f"--{k}", str(v)])
        command = command_class(None, args, unknown_args_list)
    else:
        command = command_class(None, args)

    return command.run()


def finetune(model_name_or_path: str, **kwargs) -> WorkflowOutput:
    """Fine-tune a model using LoRA or QLoRA.

    Args:
        model_name_or_path: Path to HuggingFace model
        **kwargs: All other CLI arguments supported by finetune command.
                  Includes `output_path` (defaults to "finetuned-adapter").

    Returns:
        WorkflowOutput: Contains fine-tuned adapter

    """
    kwargs["model_name_or_path"] = model_name_or_path
    return _run_unified_command(FineTuneCommand, **kwargs)


def optimize(model_name_or_path: str, **kwargs) -> WorkflowOutput:
    """Optimize the input model with comprehensive pass scheduling.

    Args:
        model_name_or_path: Path to model (file path or HuggingFace model name)
        **kwargs: All other CLI arguments supported by optimize command.
                  Includes `output_path` (defaults to "optimized-model").

    Returns:
        WorkflowOutput: Contains optimized models and metrics

    """
    kwargs["model_name_or_path"] = model_name_or_path
    return _run_unified_command(OptimizeCommand, **kwargs)


def quantize(model_name_or_path: str, **kwargs) -> WorkflowOutput:
    """Quantize a PyTorch or ONNX model.

    Args:
        model_name_or_path: Path to model file.
        **kwargs: All other CLI arguments supported by quantize command.
                  Includes `output_path` (defaults to "quantized-model").

    Returns:
        WorkflowOutput: Contains quantized model

    """
    kwargs["model_name_or_path"] = model_name_or_path
    return _run_unified_command(QuantizeCommand, **kwargs)


def capture_onnx_graph(model_name_or_path: str, **kwargs) -> WorkflowOutput:
    """Capture ONNX graph for a PyTorch model.

    Args:
        model_name_or_path: Path to PyTorch model or script
        **kwargs: All other CLI arguments supported by capture-onnx-graph command.
                  Includes `output_path` (defaults to "captured-model").

    Returns:
        WorkflowOutput: Contains captured ONNX model

    """
    kwargs["model_name_or_path"] = model_name_or_path
    return _run_unified_command(CaptureOnnxGraphCommand, **kwargs)


def generate_adapter(model_name_or_path: str, **kwargs) -> WorkflowOutput:
    """Generate adapter for an ONNX model.

    Args:
        model_name_or_path: Path to ONNX model
        **kwargs: All other CLI arguments supported by generate-adapter command.
                  Includes `output_path` (defaults to "generated-adapter").

    Returns:
        WorkflowOutput: Contains generated adapter

    """
    kwargs["model_name_or_path"] = model_name_or_path
    return _run_unified_command(GenerateAdapterCommand, **kwargs)


def tune_session_params(model_name_or_path: str, **kwargs) -> WorkflowOutput:
    """Tune ONNX Runtime session parameters for optimal performance.

    Args:
        model_name_or_path: Path to ONNX model
        **kwargs: All other CLI arguments supported by session-params-tuning command.
                  Includes `output_path` (defaults to "tuned-params").

    Returns:
        WorkflowOutput: Contains tuning results

    """
    kwargs["model_name_or_path"] = model_name_or_path
    return _run_unified_command(SessionParamsTuningCommand, **kwargs)


def generate_cost_model(model_name_or_path: str, **kwargs) -> None:
    """Generate a cost model for model splitting (HuggingFace models only).

    Args:
        model_name_or_path: Path to HuggingFace model
        **kwargs: All other CLI arguments supported by generate-cost-model command.
                  Includes `output_path` (defaults to "cost-model.csv").

    """
    kwargs["model_name_or_path"] = model_name_or_path
    _run_unified_command(GenerateCostModelCommand, **kwargs)


# Utility functions that don't necessarily produce model outputs
def convert_adapters(adapter_path: str, **kwargs) -> None:
    """Convert LoRA adapter weights to a format consumable by ONNX models.

    Args:
        adapter_path: Path to adapter weights (local folder or HuggingFace ID)
        **kwargs: All other CLI arguments supported by convert-adapters command

    """
    kwargs["adapter_path"] = adapter_path
    _run_unified_command(ConvertAdaptersCommand, **kwargs)


def extract_adapters(model_name_or_path: str, **kwargs) -> None:
    """Extract LoRA adapters from PyTorch model to separate files.

    Args:
        model_name_or_path: Path to PyTorch model (local folder or HuggingFace ID)
        **kwargs: All other CLI arguments supported by extract-adapters command

    """
    kwargs["model_name_or_path"] = model_name_or_path
    _run_unified_command(ExtractAdaptersCommand, **kwargs)
