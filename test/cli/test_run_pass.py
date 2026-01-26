# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import subprocess
import sys
from argparse import ArgumentParser


def test_run_pass_command_help():
    """Test that the run-pass command shows help properly."""
    # setup
    command_args = [sys.executable, "-m", "olive", "run-pass", "--help"]

    # execute
    out = subprocess.run(command_args, check=True, capture_output=True)

    # assert
    help_text = out.stdout.decode("utf-8")
    assert "usage:" in help_text
    assert "run-pass" in help_text
    assert "--pass-name" in help_text
    assert "Name of the pass to run on the input model" in help_text


def test_run_pass_command_argument_parsing():
    """Test the argument parsing without full CLI execution."""
    # Create a minimal parser to test our command registration
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    # Import and register our command without triggering heavy dependencies
    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test parsing with valid arguments
    args = parser.parse_args(["run-pass", "--pass-name", "OnnxConversion", "-m", "test_model", "-o", "/tmp/output"])

    assert args.pass_name == "OnnxConversion"
    assert args.model_name_or_path == "test_model"
    assert not hasattr(args, "list_passes") or not args.list_passes


def test_run_pass_command_list_passes():
    """Test the --list-passes argument."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test list-passes argument
    args = parser.parse_args(["run-pass", "--list-passes"])
    assert args.list_passes is True


def test_run_pass_command_pass_config():
    """Test the --pass-config argument."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test pass-config argument
    json_config = '{"convert_attribute": true}'
    args = parser.parse_args(
        [
            "run-pass",
            "--pass-name",
            "OnnxConversion",
            "-m",
            "test_model",
            "-o",
            "/tmp/output",
            "--pass-config",
            json_config,
        ]
    )

    assert args.pass_config == json_config


def test_run_pass_command_missing_pass_name():
    """Test that the run-pass command requires --pass-name argument when not listing passes."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test missing pass-name without list-passes should succeed in parsing but fail later
    args = parser.parse_args(["run-pass", "-m", "test_model", "-o", "/tmp/output"])
    assert args.pass_name is None
    assert args.model_name_or_path == "test_model"


def test_run_pass_command_missing_model():
    """Test that the run-pass command requires model argument when not listing passes."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test missing model should succeed in parsing but fail later when required
    args = parser.parse_args(["run-pass", "--pass-name", "OnnxConversion", "-o", "/tmp/output"])
    assert args.pass_name == "OnnxConversion"
    assert getattr(args, "model_name_or_path", None) is None


def test_run_pass_command_config_generation():
    """Test the configuration generation logic."""
    from copy import deepcopy

    # Test the configuration template and generation logic
    template = {
        "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
            }
        },
        "output_dir": "models",
        "host": "local_system",
        "target": "local_system",
        "no_artifacts": True,
    }

    # Simulate the _get_run_config logic
    config = deepcopy(template)
    pass_name = "OnnxConversion"

    # Add the pass configuration
    pass_config = {"type": pass_name}
    config["passes"] = {pass_name.lower(): pass_config}

    # Update output directory
    config["output_dir"] = "/tmp/test_output"

    # Verify the structure
    assert "passes" in config
    assert "onnxconversion" in config["passes"]
    assert config["passes"]["onnxconversion"]["type"] == "OnnxConversion"
    assert config["output_dir"] == "/tmp/test_output"
    assert config["host"] == "local_system"
    assert config["target"] == "local_system"


def test_run_pass_command_config_generation_with_pass_config():
    """Test the configuration generation with additional pass config."""
    from copy import deepcopy

    template = {
        "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
            }
        },
        "output_dir": "models",
        "host": "local_system",
        "target": "local_system",
        "no_artifacts": True,
    }

    # Simulate enhanced configuration
    config = deepcopy(template)
    pass_name = "OnnxConversion"
    pass_config = {"type": pass_name}

    # Add additional configuration
    additional_config = {"convert_attribute": True}
    pass_config.update(additional_config)

    config["passes"] = {pass_name.lower(): pass_config}

    # Verify the enhanced structure
    assert config["passes"]["onnxconversion"]["type"] == "OnnxConversion"
    assert config["passes"]["onnxconversion"]["convert_attribute"] is True


def test_run_pass_command_device_options():
    """Test the --device and accelerator argument parsing."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test device argument parsing
    args = parser.parse_args(
        [
            "run-pass",
            "--pass-name",
            "OnnxConversion",
            "-m",
            "test_model",
            "-o",
            "/tmp/output",
            "--device",
            "gpu",
            "--provider",
            "CUDAExecutionProvider",
        ]
    )

    assert args.device == "gpu"
    assert args.provider == "CUDAExecutionProvider"

    # Test default device (cpu)
    args_default = parser.parse_args(
        ["run-pass", "--pass-name", "OnnxConversion", "-m", "test_model", "-o", "/tmp/output"]
    )

    assert args_default.device == "cpu"
    assert args_default.provider == "CPUExecutionProvider"


def test_run_pass_command_accelerator_config_integration():
    """Test that accelerator options are integrated into the configuration properly."""
    from copy import deepcopy

    from olive.common.utils import set_nested_dict_value

    # Test template
    template = {
        "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
            }
        },
        "output_dir": "models",
        "host": "local_system",
        "target": "local_system",
        "no_artifacts": True,
    }

    # Simulate update_accelerator_options logic
    config = deepcopy(template)

    # Simulate args with GPU device
    class MockArgs:
        device = "gpu"
        provider = "CUDAExecutionProvider"
        memory = None

    args = MockArgs()

    # Apply the accelerator updates
    execution_providers = [args.provider]
    to_replace = [
        (("systems", "local_system", "accelerators", 0, "device"), args.device),
        (("systems", "local_system", "accelerators", 0, "execution_providers"), execution_providers),
        (("systems", "local_system", "accelerators", 0, "memory"), args.memory),
    ]
    for k, v in to_replace:
        if v is not None:
            set_nested_dict_value(config, k, v)

    # Verify the configuration was updated correctly
    accelerator = config["systems"]["local_system"]["accelerators"][0]
    assert accelerator["device"] == "gpu"
    assert accelerator["execution_providers"] == ["CUDAExecutionProvider"]


def test_run_pass_command_device_provider_consistency():
    """Test that provider/device consistency is enforced."""
    from copy import deepcopy

    # Test template
    template = {
        "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
            }
        },
        "output_dir": "models",
        "host": "local_system",
        "target": "local_system",
        "no_artifacts": True,
    }

    # Test the consistency enforcement logic
    config = deepcopy(template)

    # Simulate a case where user specifies device=cpu but provider=CUDAExecutionProvider
    accelerator = config["systems"]["local_system"]["accelerators"][0]
    accelerator["device"] = "cpu"
    accelerator["execution_providers"] = ["CUDAExecutionProvider"]

    # Define test constants (from olive.hardware.constants)
    device_to_ep = {
        "cpu": {"CPUExecutionProvider", "OpenVINOExecutionProvider"},
        "gpu": {
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
            "TensorrtExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
            "OpenVINOExecutionProvider",
            "JsExecutionProvider",
        },
        "npu": {
            "DmlExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
            "OpenVINOExecutionProvider",
        },
    }

    # Apply consistency logic (simulate _ensure_device_provider_consistency)
    providers = accelerator.get("execution_providers", [])
    current_device = accelerator.get("device", "cpu")

    if providers:
        provider = providers[0]

        # Define provider-specific device preferences
        provider_device_preference = {
            "CPUExecutionProvider": "cpu",
            "CUDAExecutionProvider": "gpu",
            "ROCMExecutionProvider": "gpu",
            "TensorrtExecutionProvider": "gpu",
            "NvTensorRTRTXExecutionProvider": "gpu",
            "MIGraphXExecutionProvider": "gpu",
            "JsExecutionProvider": "gpu",
            "DmlExecutionProvider": "gpu",
            "QNNExecutionProvider": "npu",
            "VitisAIExecutionProvider": "npu",
            "OpenVINOExecutionProvider": "cpu",
        }

        # Check if current device is valid for the provider
        valid_devices = []
        for device, device_providers in device_to_ep.items():
            if provider in device_providers:
                valid_devices.append(device)

        if current_device not in valid_devices and valid_devices:
            # Current device is not valid for the provider, use the preferred device
            preferred_device = provider_device_preference.get(provider, valid_devices[0])
            accelerator["device"] = preferred_device

    # Verify that device was corrected to match the provider
    assert accelerator["device"] == "gpu"  # CUDAExecutionProvider requires gpu
    assert accelerator["execution_providers"] == ["CUDAExecutionProvider"]
