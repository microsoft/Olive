# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.cli.launcher import main as cli_main


@pytest.mark.parametrize("console_script", [True, False])
@pytest.mark.parametrize(
    "command",
    [
        "run",
        "run-pass",
        "configure-qualcomm-sdk",
        "convert-adapters",
        "tune-session-params",
        "auto-opt",
        "optimize",
        "diffusion-lora",
    ],
)
def test_valid_command(console_script, command):
    # setup
    command_args = []
    if console_script:
        command_args.append("olive")
    else:
        command_args.extend([sys.executable, "-m", "olive"])
    if command:
        command_args.append(command)
    command_args.append("--help")

    # execute
    out = subprocess.run(command_args, check=True, capture_output=True)

    # assert
    if not console_script:
        # the help message only says `python` when running as a module
        command_args[0] = "python"
    assert f"usage: {' '.join(command_args[:-1])}" in out.stdout.decode("utf-8")


@pytest.mark.parametrize("console_script", [True, False])
def test_invalid_command(console_script):
    # setup
    command_args = []
    if console_script:
        command_args.append("olive")
    else:
        command_args.extend([sys.executable, "-m", "olive"])
    command_args.append("invalid-command")

    # execute and assert
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command_args, check=True, capture_output=True)


@pytest.mark.parametrize("deprecated_module", ["olive.workflows.run", "olive.platform_sdk.qualcomm.configure"])
def test_legacy_call(deprecated_module):
    # execute
    out = subprocess.run([sys.executable, "-m", deprecated_module, "--help"], check=True, capture_output=True)

    # assert
    assert (
        f"Running `python -m {deprecated_module}` is deprecated and might be removed in the future."
        in out.stderr.decode("utf-8")
    )


def test_unknown_args():
    # setup
    command_args = ["olive", "run", "--config", "config.json", "--unknown-arg", "-u"]

    # execute and assert
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.run(command_args, check=True, capture_output=True)

    error_message = exc_info.value.stderr.decode("utf-8")
    assert "Unknown arguments:" in error_message
    assert "--unknown-arg" in error_message
    assert "-u" in error_message


@pytest.mark.parametrize("list_required_packages", [True, False])
@pytest.mark.parametrize("tempdir", [None, "tempdir"])
@patch("olive.workflows.run")
def test_workflow_run_command(mock_run, tempdir, list_required_packages, tmp_path):
    # setup
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"key": "value"}))  # Create a dummy config file
    command_args = ["run", "--run-config", str(config_path)]
    if list_required_packages:
        command_args.append("--list_required_packages")
    if tempdir is not None:
        command_args.extend(["--tempdir", tempdir])

    # execute
    cli_main(command_args)

    # assert
    mock_run.assert_called_once_with(
        {"key": "value"}, package_config=None, tempdir=tempdir, list_required_packages=list_required_packages
    )


@patch("olive.workflows.run")
def test_workflow_run_command_with_overrides(mock_run, tmp_path):
    # setup
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"input_model": {"key": "value"}, "engine": {"log_severity_level": 3}, "output_dir": "output"})
    )
    command_args = [
        "run",
        "--run-config",
        str(config_path),
        "--model_name_or_path",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "--output_path",
        "new_output_path",
        "--log_level",
        "2",
    ]

    # execute
    cli_main(command_args)
    # assert
    mock_run.assert_called_once_with(
        {
            "input_model": {
                "type": "HfModel",
                "model_path": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                "load_kwargs": {"attn_implementation": "eager", "trust_remote_code": False},
            },
            "engine": {},
            "output_dir": str(Path("new_output_path").resolve()),
            "log_severity_level": 2,
        },
        list_required_packages=False,
        package_config=None,
        tempdir=None,
    )


@patch("olive.platform_sdk.qualcomm.configure.configure.configure")
def test_configure_qualcomm_sdk_command(mock_configure):
    # setup
    command_args = ["configure-qualcomm-sdk", "--py_version", "3.8"]

    # execute
    cli_main(command_args)

    # assert
    mock_configure.assert_called_once_with("3.8")


@patch("olive.workflows.run")
@patch("huggingface_hub.repo_exists", return_value=True)
def test_finetune_command(_, mock_run, tmp_path):
    # setup
    output_dir = tmp_path / "output_dir"

    # setup
    model_id = "dummy-model-id"
    command_args = [
        "finetune",
        "-m",
        model_id,
        "-d",
        "dummy_dataset",
        "--text_field",
        "dummy_text_field",
        "-o",
        str(output_dir),
    ]

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == model_id
    assert mock_run.call_count == 1


@patch("olive.workflows.run")
@patch("olive.model.handler.diffusers.is_valid_diffusers_model", return_value=True)
def test_diffusion_lora_command(_, mock_run, tmp_path):
    # setup
    output_dir = tmp_path / "output_dir"
    data_dir = tmp_path / "train_images"
    data_dir.mkdir()

    # Create dummy training images
    from PIL import Image

    for i in range(2):
        img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
        img.save(data_dir / f"image_{i}.png")
        (data_dir / f"image_{i}.txt").write_text(f"a test image {i}")

    model_id = "runwayml/stable-diffusion-v1-5"
    command_args = [
        "diffusion-lora",
        "-m",
        model_id,
        "-d",
        str(data_dir),
        "-o",
        str(output_dir),
        "--model_type",
        "sd15",
        "--max_train_steps",
        "1",
    ]

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == model_id
    assert config["input_model"]["type"] == "DiffusersModel"
    assert "sd_lora" in config["passes"]
    assert mock_run.call_count == 1


def test_session_params_tuning_command(tmp_path):
    from test.utils import ONNX_MODEL_PATH

    # some directories
    output_dir = tmp_path / "output_dir"

    # setup
    command_args = [
        "tune-session-params",
        "-m",
        str(ONNX_MODEL_PATH),
        "--output_path",
        str(output_dir),
        "--providers_list",
        "CPUExecutionProvider",
    ]

    # execute
    # run in subprocess to avoid affecting other tests
    out = subprocess.run(["olive", *command_args], check=True, capture_output=True)

    # assert
    assert f"Inference session parameters are saved to {output_dir}" in out.stdout.decode("utf-8")
    with open(output_dir / "cpu-cpu.json") as f:
        infer_settings = json.load(f)
        assert infer_settings["execution_provider"] == ["CPUExecutionProvider"]
        assert infer_settings.keys() >= {"provider_options", "session_options"}


@patch("olive.workflows.run")
@patch("huggingface_hub.repo_exists", return_value=True)
@pytest.mark.parametrize("use_model_builder", [True, False])
def test_capture_onnx_command(_, mock_run, use_model_builder, tmp_path):
    # setup
    output_dir = tmp_path / "output_dir"
    model_id = "dummy-model-id"
    command_args = [
        "capture-onnx-graph",
        "-m",
        model_id,
        "-o",
        str(output_dir),
    ]

    if use_model_builder:
        command_args.extend(["--use_model_builder", "--precision", "int4"])

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == model_id
    assert "m" in config["passes"] if use_model_builder else "c" in config["passes"]
    assert mock_run.call_count == 1


@patch("olive.workflows.run")
@patch("huggingface_hub.repo_exists", return_value=True)
@pytest.mark.parametrize("use_model_builder", [True, False])
def test_capture_onnx_command_fix_shape(_, mock_run, use_model_builder, tmp_path):
    # setup
    output_dir = tmp_path / "output_dir"
    model_id = "dummy-model-id"
    fixed_param_dict = {"batch_size": 1, "max_seq_len": 512}
    fixed_param_dict_str = "batch_size=1,max_seq_len=512"
    command_args = [
        "capture-onnx-graph",
        "-m",
        model_id,
        "--fixed_param_dict",
        fixed_param_dict_str,
        "-o",
        str(output_dir),
    ]

    if use_model_builder:
        command_args.extend(["--use_model_builder", "--precision", "int4"])

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == model_id
    assert "m" in config["passes"] if use_model_builder else "c" in config["passes"]
    assert "f" in config["passes"]
    assert config["passes"]["f"]["dim_param"] == list(fixed_param_dict.keys())
    assert config["passes"]["f"]["dim_value"] == list(fixed_param_dict.values())
    assert mock_run.call_count == 1


@patch("olive.cli.shared_cache.AzureContainerClientFactory")
def test_shared_cache_command(mock_AzureContainerClientFactory):
    # setup
    mock_factory_instance = mock_AzureContainerClientFactory.return_value
    command_args = [
        "shared-cache",
        "--delete",
        "--account",
        "account",
        "--container",
        "container",
        "--model_hash",
        "model_hash",
    ]

    # execute
    cli_main(command_args)

    # assert
    mock_factory_instance.delete_blob.assert_called_once_with("model_hash")


@patch("builtins.input", side_effect=lambda _: "y")
@patch("olive.cli.shared_cache.AzureContainerClientFactory")
def test_shared_cache_delete_all_with_confirmation(mock_AzureContainerClientFactory, mock_input):
    # setup
    mock_factory_instance = mock_AzureContainerClientFactory.return_value
    command_args = [
        "shared-cache",
        "--delete",
        "--account",
        "account",
        "--container",
        "container",
        "--all",
    ]

    # execute
    cli_main(command_args)

    # assert
    mock_input.assert_called_once_with("Are you sure you want to delete all cache? (y/n): ")
    mock_factory_instance.delete_all.assert_called_once()


@pytest.mark.parametrize("algorithm_name", ["awq", "gptq", "lpbq", "seqmse", "adaround"])
@patch("olive.workflows.run")
@patch("huggingface_hub.repo_exists")
def test_quantize_command(mock_repo_exists, mock_run, algorithm_name, tmp_path):
    from test.utils import ONNX_MODEL_PATH

    # setup
    output_dir = tmp_path / "output_dir"

    # setup
    command_args = [
        "quantize",
        "--algorithm",
        algorithm_name,
        "-o",
        str(output_dir),
    ]

    model_name = "dummy_model"
    if algorithm_name == "gptq":
        command_args += ["-d", "dummy_dataset"]
        command_args += ["--implementation", "autogptq"]
    if algorithm_name == "awq":
        command_args += ["--implementation", "awq"]
    if algorithm_name in {"lpbq", "seqmse", "adaround"}:
        model_name = str(ONNX_MODEL_PATH)
        command_args += ["-d", "dummy_dataset"]
        command_args += ["--implementation", "aimet"]

    command_args += ["-m", model_name]

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == model_name
    assert mock_run.call_count == 1


# TODO(anyone): resolve CI package installation issues later and then re-enable this unit test
# @patch("huggingface_hub.repo_exists", return_value=True)
# def test_extract_adapters_command_from_transformers_model(mock_repo_exists, tmp_path):
#     # setup
#     output_dir = tmp_path / "output_dir"
#     cache_dir = tmp_path / "cache_dir"
#     model_id = "microsoft/Phi-4-multimodal-instruct"
#     command_args = [
#         "extract-adapters",
#         "-m",
#         model_id,
#         "-o",
#         str(output_dir),
#         "-f",
#         "onnx_adapter",
#         "--cache_dir",
#         str(cache_dir),
#     ]

#     # execute
#     cli_main(command_args)

#     assert (output_dir / "vision.onnx_adapter").exists()
#     assert (output_dir / "speech.onnx_adapter").exists()


@patch("huggingface_hub.repo_exists", return_value=True)
def test_extract_adapters_command_from_peft_model(mock_repo_exists, tmp_path):
    # setup
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    cache_dir = tmp_path / "cache_dir"
    base_model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", cache_dir=str(cache_dir)
    )
    base_model.generation_config.pad_token_id = 0

    speech_config = LoraConfig(
        r=2,
        lora_alpha=4.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, speech_config, adapter_name="speech")

    vision_config = LoraConfig(
        r=4,
        lora_alpha=12.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    peft_model.add_adapter("vision", vision_config)

    peft_model.save_pretrained(str(tmp_path))
    peft_model.base_model.save_pretrained(str(tmp_path))

    output_dir = tmp_path / "output_dir"
    command_args = [
        "extract-adapters",
        "-m",
        str(tmp_path),
        "-o",
        str(output_dir),
        "-f",
        "onnx_adapter",
        "--cache_dir",
        str(cache_dir),
    ]

    # execute
    cli_main(command_args)

    assert (output_dir / "vision.onnx_adapter").exists()
    assert (output_dir / "speech.onnx_adapter").exists()


# Test for ConvertAdaptersCommand is added as part of test/passes/onnx/test_extract_adapters.py


@patch("olive.workflows.run")
@patch("huggingface_hub.repo_exists")
def test_optimize_cli_pass_list(mock_repo_exists, mock_run, tmp_path):
    # setup
    output_dir = "output_dir"

    test_list = [
        [
            "optimize",
            (
                "--precision int4 --act_precision int8 --provider QNNExecutionProvider --num_split 4 --use_qdq_format "
                '--block_size -1 --surgeries "RemoveRopeMultiCache,AttentionMaskToSequenceLengths"'
            ),
            (
                "QuaRot, Gptq, CaptureSplitInfo, ModelBuilder, MatMulNBitsToQDQ, GraphSurgeries, "
                "OnnxStaticQuantization, SplitModel, StaticLLM"
            ),
        ],
        [
            "optimize",
            (
                " --precision int4 --act_precision int16 --provider VitisAIExecutionProvider  --num_split 4 "
                '--use_qdq_format --surgeries "RemoveRopeMultiCache,AttentionMaskToSequenceLengths"  --block_size -1'
            ),
            (
                "QuaRot, Gptq, CaptureSplitInfo, ModelBuilder, MatMulNBitsToQDQ, GraphSurgeries, "
                "OnnxStaticQuantization, VitisAIAddMetaData, SplitModel, StaticLLM"
            ),
        ],
        [
            "optimize",
            "--precision int4 --act_precision int16 --provider OpenVINOExecutionProvider  --device gpu",
            "OpenVINOOptimumConversion, OpenVINOIoUpdate, OpenVINOEncapsulation",
        ],
        [
            "optimize",
            "-t text-classification --precision int8 --exporter torchscript_exporter",
            "OnnxConversion, OnnxPeepholeOptimizer, OrtTransformersOptimization, OnnxStaticQuantization",
        ],
        [
            "optimize",
            (
                "-t text-classification --precision int8 --exporter torchscript_exporter --provider"
                ' QNNExecutionProvider --device npu --dim_param "batch_size,sequence_length" --dim_value 1,128'
            ),
            (
                "OnnxConversion, DynamicToFixedShape, OnnxPeepholeOptimizer, OrtTransformersOptimization, "
                "OnnxStaticQuantization, StaticLLM"
            ),
        ],
        [
            "optimize",
            "-t text-classification --precision fp16 --exporter torchscript_exporter --provider CUDAExecutionProvider",
            "OnnxConversion, OnnxPeepholeOptimizer, OrtTransformersOptimization, OnnxFloatToFloat16",
        ],
        [
            "optimize",
            (
                "-t text-classification --precision fp16 --exporter torchscript_exporter --provider"
                " NvTensorRTRTXExecutionProvider --device gpu"
            ),
            "OnnxConversion, OnnxPeepholeOptimizer, OnnxFloatToFloat16",
        ],
    ]

    for t in test_list:
        # setup
        command_args = [
            t[0],
            "-m",
            "dummy_model",
            "--dry_run",
            "-o",
            str(output_dir),
        ]

        command_args += [item.strip() for item in t[1].split()]

        # execute
        cli_main(command_args)

        json_file_path = "output_dir/config.json"
        with open(json_file_path) as file:
            data = json.load(file)

        passes = data.get("passes", {})
        pass_list = [k[1]["type"] for k in passes.items()]

        assert pass_list == [item.strip() for item in t[2].split(",")]


@patch("olive.workflows.run")
@patch("huggingface_hub.repo_exists", return_value=True)
def test_benchmark_command(_, mock_run, tmp_path):
    # setup
    output_dir = tmp_path / "output_dir"
    model_id = "dummy-model-id"
    command_args = [
        "benchmark",
        "-m",
        model_id,
        "-o",
        str(output_dir),
        "--tasks",
        "arc_easy",
        "helloswag",
        "--device",
        "gpu",
        "--batch_size",
        "8",
        "--max_length",
        "1024",
        "--limit",
        "16",
    ]

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == model_id
    assert config["evaluators"]["evaluator"]["tasks"] == ["arc_easy", "helloswag"]
    assert config["evaluators"]["evaluator"]["batch_size"] == 8
    assert config["evaluators"]["evaluator"]["max_length"] == 1024
    assert config["evaluators"]["evaluator"]["limit"] == 16
    assert mock_run.call_count == 1
