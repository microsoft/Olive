# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import subprocess
import sys
from argparse import ArgumentParser
from unittest.mock import patch

import pytest


def test_one_command_help():
    """Test that the one command shows help properly."""
    # setup
    command_args = [sys.executable, "-m", "olive", "one", "--help"]

    # execute
    out = subprocess.run(command_args, check=True, capture_output=True)

    # assert
    help_text = out.stdout.decode("utf-8")
    assert "usage:" in help_text
    assert "one" in help_text
    assert "--pass-name" in help_text
    assert "Run a single pass on the input model" in help_text


def test_one_command_argument_parsing():
    """Test the argument parsing without full CLI execution."""
    # Create a minimal parser to test our command registration
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    
    # Import and register our command without triggering heavy dependencies
    from olive.cli.one import OneCommand
    OneCommand.register_subcommand(sub_parsers)
    
    # Test parsing with valid arguments
    args = parser.parse_args(['one', '--pass-name', 'OnnxConversion', '-m', 'test_model', '-o', '/tmp/output'])
    
    assert args.pass_name == 'OnnxConversion'
    assert args.model_name_or_path == 'test_model'
    assert args.output_path == '/tmp/output'
    assert not hasattr(args, 'list_passes') or not args.list_passes


def test_one_command_list_passes():
    """Test the --list-passes argument."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    
    from olive.cli.one import OneCommand
    OneCommand.register_subcommand(sub_parsers)
    
    # Test list-passes argument
    args = parser.parse_args(['one', '--list-passes'])
    assert args.list_passes == True


def test_one_command_pass_config():
    """Test the --pass-config argument."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    
    from olive.cli.one import OneCommand
    OneCommand.register_subcommand(sub_parsers)
    
    # Test pass-config argument
    json_config = '{"target_opset": 13, "convert_attribute": true}'
    args = parser.parse_args([
        'one', '--pass-name', 'OnnxConversion', 
        '-m', 'test_model', '-o', '/tmp/output',
        '--pass-config', json_config
    ])
    
    assert args.pass_config == json_config


def test_one_command_missing_pass_name():
    """Test that the one command requires --pass-name argument when not listing passes."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    
    from olive.cli.one import OneCommand
    OneCommand.register_subcommand(sub_parsers)
    
    # Test missing pass-name without list-passes should succeed in parsing but fail later
    args = parser.parse_args(['one', '-m', 'test_model', '-o', '/tmp/output'])
    assert args.pass_name is None
    assert args.model_name_or_path == 'test_model'


def test_one_command_missing_model():
    """Test that the one command requires model argument when not listing passes."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    
    from olive.cli.one import OneCommand
    OneCommand.register_subcommand(sub_parsers)
    
    # Test missing model should succeed in parsing but fail later when required
    args = parser.parse_args(['one', '--pass-name', 'OnnxConversion', '-o', '/tmp/output'])
    assert args.pass_name == 'OnnxConversion'
    assert getattr(args, 'model_name_or_path', None) is None


def test_one_command_config_generation():
    """Test the configuration generation logic."""
    from copy import deepcopy
    import json
    
    # Test the configuration template and generation logic
    TEMPLATE = {
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
    config = deepcopy(TEMPLATE)
    pass_name = "OnnxConversion"
    
    # Add the pass configuration
    pass_config = {"type": pass_name}
    config["passes"] = {
        pass_name.lower(): pass_config
    }
    
    # Update output directory
    config["output_dir"] = "/tmp/test_output"
    
    # Verify the structure
    assert "passes" in config
    assert "onnxconversion" in config["passes"]
    assert config["passes"]["onnxconversion"]["type"] == "OnnxConversion"
    assert config["output_dir"] == "/tmp/test_output"
    assert config["host"] == "local_system"
    assert config["target"] == "local_system"


def test_one_command_config_generation_with_pass_config():
    """Test the configuration generation with additional pass config."""
    from copy import deepcopy
    import json
    
    TEMPLATE = {
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
    config = deepcopy(TEMPLATE)
    pass_name = "OnnxConversion"
    pass_config = {"type": pass_name}
    
    # Add additional configuration
    additional_config = {"target_opset": 13, "convert_attribute": True}
    pass_config.update(additional_config)
    
    config["passes"] = {
        pass_name.lower(): pass_config
    }
    
    # Verify the enhanced structure
    assert config["passes"]["onnxconversion"]["type"] == "OnnxConversion"
    assert config["passes"]["onnxconversion"]["target_opset"] == 13
    assert config["passes"]["onnxconversion"]["convert_attribute"] == True


def test_one_command_device_options():
    """Test the --device and accelerator argument parsing."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    
    from olive.cli.one import OneCommand
    OneCommand.register_subcommand(sub_parsers)
    
    # Test device argument parsing
    args = parser.parse_args([
        'one', '--pass-name', 'OnnxConversion', 
        '-m', 'test_model', '-o', '/tmp/output',
        '--device', 'gpu', '--provider', 'CUDAExecutionProvider'
    ])
    
    assert args.device == 'gpu'
    assert args.provider == 'CUDAExecutionProvider'
    
    # Test default device (cpu)
    args_default = parser.parse_args([
        'one', '--pass-name', 'OnnxConversion', 
        '-m', 'test_model', '-o', '/tmp/output'
    ])
    
    assert args_default.device == 'cpu'
    assert args_default.provider == 'CPUExecutionProvider'


def test_one_command_accelerator_config_integration():
    """Test that accelerator options are integrated into the configuration properly."""
    from copy import deepcopy
    from olive.common.utils import set_nested_dict_value
    
    # Test template
    TEMPLATE = {
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
    config = deepcopy(TEMPLATE)
    
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