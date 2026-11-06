# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import subprocess
import sys
from argparse import ArgumentParser, ArgumentTypeError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from olive.cli.launcher import main as cli_main
from olive.cli.pass_search import OLIVE_AAS_RUN_URL, PassSearchCommand, _hex_string

# pylint: disable=W0212


def _build_parser():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()
    PassSearchCommand.register_subcommand(sub_parsers)
    return parser


# --------------------------------------------------------------------------- #
# Help / registration                                                         #
# --------------------------------------------------------------------------- #


def test_search_command_help():
    # setup
    command_args = [sys.executable, "-m", "olive", "pass-search", "--help"]

    # execute
    out = subprocess.run(command_args, check=True, capture_output=True)

    # assert
    help_text = out.stdout.decode("utf-8")
    assert "usage:" in help_text
    assert "pass-search" in help_text
    assert "--run-config" in help_text
    assert "--passes" in help_text


# --------------------------------------------------------------------------- #
# _hex_string validation                                                      #
# --------------------------------------------------------------------------- #


def test_hex_string_accepts_valid_32_char_hex():
    value = "0123456789abcdef0123456789ABCDEF"
    assert _hex_string(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0123456789abcdef",  # too short
        "0123456789abcdef0123456789abcdef0",  # too long
        "0123456789abcdef0123456789abcdeg",  # invalid char 'g'
        "0123456789abcdef 123456789abcdef",  # space
    ],
)
def test_hex_string_rejects_invalid_values(value):
    with pytest.raises(ArgumentTypeError):
        _hex_string(value)


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def test_search_mutually_exclusive_mode_required():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pass-search"])


def test_search_run_config_and_model_are_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pass-search", "--run-config", "config.json", "-m", "some-model"])


def test_search_parses_build_mode_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "pass-search",
            "-m",
            "microsoft/phi-2",
            "--passes",
            "OnnxConversion",
            "OnnxQuantization",
            "--device",
            "gpu",
            "--provider",
            "CUDAExecutionProvider",
            "--output_path",
            "out",
        ]
    )
    assert args.model_name_or_path == "microsoft/phi-2"
    assert args.run_config is None
    assert args.passes == ["OnnxConversion", "OnnxQuantization"]
    assert args.device == "gpu"
    assert args.provider == "CUDAExecutionProvider"
    assert args.output_path == "out"


def test_search_parses_config_mode_args():
    parser = _build_parser()
    args = parser.parse_args(["pass-search", "--config", "config.json"])
    assert args.run_config == "config.json"
    assert args.model_name_or_path is None


def test_search_rejects_invalid_subscription_key():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pass-search", "-m", "model", "--az_apim_subscription_key", "not-hex"])


def test_search_parses_search_strategy_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "pass-search",
            "-m",
            "model",
            "--execution-order",
            "pass-by-pass",
            "--sampler",
            "tpe",
            "--max-iter",
            "10",
            "--max-time",
            "60",
            "--output-model-num",
            "3",
            "--stop-when-goals-met",
            "--include-pass-params",
        ]
    )
    assert args.execution_order == "pass-by-pass"
    assert args.sampler == "tpe"
    assert args.max_iter == 10
    assert args.max_time == 60
    assert args.output_model_num == 3
    assert args.stop_when_goals_met is True
    assert args.include_pass_params is True


def test_search_rejects_invalid_sampler():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["pass-search", "-m", "model", "--sampler", "invalid"])


# --------------------------------------------------------------------------- #
# _parse_pass_groups                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # single unbracketed pass -> one single-pass group
        (["OnnxConversion"], [["OnnxConversion"]]),
        # multiple unbracketed passes -> each its own group
        (["OnnxConversion", "OnnxQuantization"], [["OnnxConversion"], ["OnnxQuantization"]]),
        # single bracketed multi-pass group
        (["[OnnxConversion, OnnxQuantization]"], [["OnnxConversion", "OnnxQuantization"]]),
        # two bracketed groups
        (
            ["[OnnxConversion, OrtTransformersOptimization]", "[OnnxConversion, OnnxQuantization]"],
            [["OnnxConversion", "OrtTransformersOptimization"], ["OnnxConversion", "OnnxQuantization"]],
        ),
        # mixed bracketed and unbracketed, order preserved
        (
            [
                "[OnnxConversion, OrtTransformersOptimization]",
                "OnnxQuantization",
                "[OnnxConversion, OnnxFloatToFloat16]",
            ],
            [
                ["OnnxConversion", "OrtTransformersOptimization"],
                ["OnnxQuantization"],
                ["OnnxConversion", "OnnxFloatToFloat16"],
            ],
        ),
        # space-separated passes inside brackets
        (["[OnnxConversion OnnxQuantization]"], [["OnnxConversion", "OnnxQuantization"]]),
        # either/or slot inside a bracketed pipeline
        (
            ["[OnnxConversion, (OnnxQuantization | OnnxStaticQuantization)]"],
            [["OnnxConversion", ["OnnxQuantization", "OnnxStaticQuantization"]]],
        ),
        # either/or slot with more than two alternatives
        (
            ["[SMP, (RTN | GPTQ | AWQ)]"],
            [["SMP", ["RTN", "GPTQ", "AWQ"]]],
        ),
        # commas also separate alternatives within parentheses
        (
            ["[SMP, (RTN, GPTQ)]"],
            [["SMP", ["RTN", "GPTQ"]]],
        ),
        # bare (top-level) either/or slot -> its own single-slot pipeline
        (
            ["(RTN | GPTQ)"],
            [[["RTN", "GPTQ"]]],
        ),
        # a single alternative degenerates to a plain single-pass slot
        (
            ["[SMP, (GPTQ)]"],
            [["SMP", "GPTQ"]],
        ),
        # multiple pipelines mixing sequential and either/or slots
        (
            ["[SMP, (RTN | GPTQ)]", "[SMP, GPTQ]", "[QuaRot, GPTQ]", "[SMP, QuaRot, (RTN | GPTQ)]"],
            [
                ["SMP", ["RTN", "GPTQ"]],
                ["SMP", "GPTQ"],
                ["QuaRot", "GPTQ"],
                ["SMP", "QuaRot", ["RTN", "GPTQ"]],
            ],
        ),
    ],
)
def test_parse_pass_groups(raw, expected):
    assert PassSearchCommand._parse_pass_groups(raw) == expected


def test_parse_pass_groups_empty_raises():
    with pytest.raises(ValueError, match="No valid pass groups"):
        PassSearchCommand._parse_pass_groups([""])


# --------------------------------------------------------------------------- #
# _build_search_strategy_overrides                                            #
# --------------------------------------------------------------------------- #


def test_build_search_strategy_overrides_all_set():
    args = SimpleNamespace(
        execution_order="joint",
        sampler="tpe",
        max_iter=10,
        max_time=60,
        output_model_num=2,
        stop_when_goals_met=True,
        include_pass_params=True,
    )
    overrides = PassSearchCommand._build_search_strategy_overrides(args)
    assert overrides == {
        "execution_order": "joint",
        "sampler": "tpe",
        "max_iter": 10,
        "max_time": 60,
        "output_model_num": 2,
        "stop_when_goals_met": True,
        "include_pass_params": True,
    }


def test_build_search_strategy_overrides_none_set():
    args = SimpleNamespace(
        execution_order=None,
        sampler=None,
        max_iter=None,
        max_time=None,
        output_model_num=None,
        stop_when_goals_met=False,
        include_pass_params=None,
    )
    assert not PassSearchCommand._build_search_strategy_overrides(args)


def test_build_search_strategy_overrides_include_pass_params_false():
    args = SimpleNamespace(
        execution_order=None,
        sampler=None,
        max_iter=None,
        max_time=None,
        output_model_num=None,
        stop_when_goals_met=False,
        include_pass_params=False,
    )
    assert PassSearchCommand._build_search_strategy_overrides(args) == {"include_pass_params": False}


# --------------------------------------------------------------------------- #
# _apply_search_strategy                                                      #
# --------------------------------------------------------------------------- #


def test_apply_search_strategy_no_overrides_leaves_config_untouched():
    run_config = {"input_model": {}}
    PassSearchCommand._apply_search_strategy(run_config, {})
    assert run_config == {"input_model": {}}


def test_apply_search_strategy_creates_engine_section():
    run_config = {}
    PassSearchCommand._apply_search_strategy(run_config, {"sampler": "tpe"})
    assert run_config["engine"]["search_strategy"] == {"sampler": "tpe"}


def test_apply_search_strategy_merges_existing():
    run_config = {"engine": {"search_strategy": {"max_iter": 5, "sampler": "random"}}}
    PassSearchCommand._apply_search_strategy(run_config, {"sampler": "tpe"})
    assert run_config["engine"]["search_strategy"] == {"max_iter": 5, "sampler": "tpe"}


def test_apply_search_strategy_replaces_non_dict_existing():
    run_config = {"engine": {"search_strategy": "not-a-dict"}}
    PassSearchCommand._apply_search_strategy(run_config, {"sampler": "tpe"})
    assert run_config["engine"]["search_strategy"] == {"sampler": "tpe"}


# --------------------------------------------------------------------------- #
# _build_run_config / _add_evaluator / _add_data_config                       #
# --------------------------------------------------------------------------- #


def _make_command(**overrides):
    """Create a PassSearchCommand instance with a fake args namespace."""
    defaults = {
        "model_name_or_path": "microsoft/phi-2",
        "task": None,
        "trust_remote_code": False,
        "device": "cpu",
        "provider": "CPUExecutionProvider",
        "eval_tasks": None,
        "eval_batch_size": 1,
        "eval_max_length": 1024,
        "eval_limit": 1,
        "eval_backend": "auto",
        "data_name": None,
        "log_level": None,
    }
    defaults.update(overrides)
    command = PassSearchCommand.__new__(PassSearchCommand)
    command.args = SimpleNamespace(**defaults)
    return command


def test_build_run_config_basic():
    command = _make_command()
    config = command._build_run_config(["OnnxConversion"], "out/config_00")
    assert config["input_model"] == {"type": "HfModel", "model_path": "microsoft/phi-2"}
    assert config["passes"] == {"onnxconversion": {"type": "OnnxConversion"}}
    assert config["target"] == "local_system"
    assert config["output_dir"] == "out/config_00"
    assert config["no_artifacts"] is True
    assert config["systems"]["local_system"]["accelerators"][0] == {
        "device": "cpu",
        "execution_providers": ["CPUExecutionProvider"],
    }
    # no evaluator / data config / log level by default
    assert "evaluator" not in config
    assert "data_configs" not in config
    assert "log_severity_level" not in config


def test_build_run_config_with_task_and_trust_remote_code():
    command = _make_command(task="text-generation", trust_remote_code=True)
    config = command._build_run_config(["OnnxConversion"], "out")
    assert config["input_model"]["task"] == "text-generation"
    assert config["input_model"]["load_kwargs"] == {"trust_remote_code": True}


def test_build_run_config_preserves_pass_order():
    command = _make_command()
    config = command._build_run_config(["OnnxConversion", "OrtTransformersOptimization"], "out")
    assert list(config["passes"].keys()) == ["onnxconversion", "orttransformersoptimization"]


def test_build_run_config_with_alternatives_maps_pass_to_list():
    command = _make_command()
    config = command._build_run_config(["SMP", ["RTN", "GPTQ"]], "out")
    assert config["passes"]["smp"] == {"type": "SMP"}
    assert config["passes"]["rtn_or_gptq"] == [{"type": "RTN"}, {"type": "GPTQ"}]
    assert list(config["passes"].keys()) == ["smp", "rtn_or_gptq"]


def test_build_run_config_with_alternatives_injects_search_strategy():
    command = _make_command()
    config = command._build_run_config(["SMP", ["RTN", "GPTQ"]], "out")
    # Either/or slots are explored via the always-on search strategy.
    assert config["engine"]["search_strategy"] == {"execution_order": "joint", "sampler": "sequential"}


def test_build_run_config_always_injects_search_strategy():
    command = _make_command()
    # pass-search is a search command, so a search strategy is added even without alternatives.
    config = command._build_run_config(["OnnxConversion", "OnnxQuantization"], "out")
    assert config["engine"]["search_strategy"] == {"execution_order": "joint", "sampler": "sequential"}


def test_build_run_config_sets_log_level():
    command = _make_command(log_level=2)
    config = command._build_run_config(["OnnxConversion"], "out")
    assert config["log_severity_level"] == 2


def test_add_evaluator_when_eval_tasks_provided():
    command = _make_command(eval_tasks=["hellaswag", "winogrande"], eval_limit=0.1, device="gpu", eval_backend="ort")
    config = command._build_run_config(["OnnxConversion"], "out")
    evaluator = config["evaluators"]["evaluator"]
    assert evaluator["type"] == "LMEvaluator"
    assert evaluator["tasks"] == ["hellaswag", "winogrande"]
    assert evaluator["device"] == "gpu"
    assert evaluator["limit"] == 0.1
    assert evaluator["model_class"] == "ort"
    assert config["evaluator"] == "evaluator"
    assert config["host"] == "local_system"


def test_add_evaluator_auto_backend_has_no_model_class():
    command = _make_command(eval_tasks=["hellaswag"], eval_backend="auto")
    config = command._build_run_config(["OnnxConversion"], "out")
    assert "model_class" not in config["evaluators"]["evaluator"]


def test_add_data_config_when_data_name_provided():
    command = _make_command(data_name="wikitext")
    with patch("olive.cli.pass_search.update_dataset_options") as mock_update:
        config = command._build_run_config(["OnnxStaticQuantization"], "out")
    assert config["data_configs"][0]["name"] == "default_data_config"
    assert config["data_configs"][0]["type"] == "HuggingfaceContainer"
    mock_update.assert_called_once()


def test_add_data_config_absent_without_data_name():
    command = _make_command(data_name=None)
    config = command._build_run_config(["OnnxConversion"], "out")
    assert "data_configs" not in config


# --------------------------------------------------------------------------- #
# _submit_remote_job                                                          #
# --------------------------------------------------------------------------- #


def test_submit_remote_job_returns_op_id_from_string():
    mock_response = MagicMock()
    mock_response.json.return_value = "op_12345"
    with patch("requests.post", return_value=mock_response) as mock_post:
        op_id = PassSearchCommand._submit_remote_job({"key": "value"}, "a" * 32)
    assert op_id == "op_12345"
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Ocp-Apim-Subscription-Key"] == "a" * 32
    assert kwargs["json"] == {"run_config": {"key": "value"}}


def test_submit_remote_job_returns_op_id_from_json():
    mock_response = MagicMock()
    mock_response.json.return_value = {"op_id": "op_67890"}
    with patch("requests.post", return_value=mock_response):
        op_id = PassSearchCommand._submit_remote_job({"key": "value"}, "b" * 32)
    assert op_id == "op_67890"


def test_submit_remote_job_strips_quotes():
    mock_response = MagicMock()
    mock_response.json.return_value = '"op_quoted"'
    with patch("requests.post", return_value=mock_response):
        op_id = PassSearchCommand._submit_remote_job({}, "c" * 32)
    assert op_id == "op_quoted"


def test_submit_remote_job_raises_when_no_op_id():
    mock_response = MagicMock()
    mock_response.json.return_value = {"unexpected": "field"}
    mock_response.text = '{"unexpected": "field"}'
    with patch("requests.post", return_value=mock_response), pytest.raises(RuntimeError, match="Unexpected response"):
        PassSearchCommand._submit_remote_job({}, "d" * 32)


def test_submit_remote_job_uses_correct_url():
    mock_response = MagicMock()
    mock_response.json.return_value = "op_1"
    with patch("requests.post", return_value=mock_response) as mock_post:
        PassSearchCommand._submit_remote_job({}, "e" * 32)
    args, _ = mock_post.call_args
    assert args[0] == OLIVE_AAS_RUN_URL


# --------------------------------------------------------------------------- #
# _count_search_points                                                        #
# --------------------------------------------------------------------------- #


def test_count_search_points_returns_none_when_no_passes():
    assert PassSearchCommand._count_search_points({"passes": {}}) is None
    assert PassSearchCommand._count_search_points({}) is None


def test_count_search_points_returns_one_when_no_search_strategy():
    run_config = {"passes": {"onnxconversion": {"type": "OnnxConversion"}}}
    assert PassSearchCommand._count_search_points(run_config) == 1


def test_count_search_points_returns_none_for_unknown_pass_type():
    run_config = {
        "passes": {"mystery": {"type": "NotARealPassType"}},
        "engine": {"search_strategy": {"execution_order": "joint", "sampler": "sequential"}},
    }
    assert PassSearchCommand._count_search_points(run_config) is None


def test_count_search_points_includes_pass_params_when_enabled():
    # Gptq has multiple searchable parameters (bits, group_size, sym, lm_head from
    # get_quantizer_config plus damp_percent). With include_pass_params=True the joint
    # search space is their product.
    run_config = {
        "passes": {"gptq": {"type": "Gptq"}},
        "engine": {
            "search_strategy": {
                "execution_order": "joint",
                "sampler": "sequential",
                "include_pass_params": True,
            }
        },
    }
    # bits(3) * group_size(5) * sym(2) * lm_head(2) * damp_percent(3) = 180
    assert PassSearchCommand._count_search_points(run_config) == 180


def test_count_search_points_excludes_pass_params_by_default():
    # include_pass_params defaults to False, so per-pass parameters are excluded from the
    # search space. A single pass config then yields exactly one search point regardless of
    # how many searchable parameters the pass defines.
    run_config = {
        "passes": {"gptq": {"type": "Gptq"}},
        "engine": {"search_strategy": {"execution_order": "joint", "sampler": "sequential"}},
    }
    assert PassSearchCommand._count_search_points(run_config) == 1


def test_count_search_points_excludes_pass_params_when_disabled():
    run_config = {
        "passes": {"gptq": {"type": "Gptq"}},
        "engine": {
            "search_strategy": {
                "execution_order": "joint",
                "sampler": "sequential",
                "include_pass_params": False,
            }
        },
    }
    assert PassSearchCommand._count_search_points(run_config) == 1


def test_count_search_points_for_multiple_pass_groups():
    # Mirrors the build-mode invocation `--passes [SMP] [RTN, GPTQ]`, which generates two
    # separate configs (one per bracketed group). Each config's search space is counted
    # independently. Within the [RTN, GPTQ] group the joint space is the product of the two
    # passes' searchable parameter combinations.
    strategy = {"execution_order": "joint", "sampler": "sequential", "include_pass_params": True}

    smp_group = {
        "passes": {"selectivemixedprecision": {"type": "SelectiveMixedPrecision"}},
        "engine": {"search_strategy": strategy},
    }
    # SMP: algorithm(8) * bits(3) * group_size(5) * sym(2) * high_bits(2)
    #      * high_group_size(6) * high_sym(3) * ratio(4) = 34560
    assert PassSearchCommand._count_search_points(smp_group) == 34560

    rtn_gptq_group = {
        "passes": {"rtn": {"type": "Rtn"}, "gptq": {"type": "Gptq"}},
        "engine": {"search_strategy": strategy},
    }
    # Rtn: bits(3) * group_size(5) * sym(2) * lm_head(2) = 60
    # Gptq: bits(3) * group_size(5) * sym(2) * lm_head(2) * damp_percent(3) = 180
    # Joint search space for the group = 60 * 180 = 10800
    assert PassSearchCommand._count_search_points(rtn_gptq_group) == 10800


def test_count_search_points_with_alternative_pass_configs():
    # A pass name can map to a list of alternative configs. The engine wraps them in a
    # Categorical, so that pass contributes the *sum* of each alternative's search-space size.
    run_config = {
        "passes": {
            "smp": {"type": "SelectiveMixedPrecision"},
            "quantize": [{"type": "Rtn"}, {"type": "Gptq"}],
        },
        "engine": {
            "search_strategy": {
                "execution_order": "joint",
                "sampler": "sequential",
                "include_pass_params": True,
            }
        },
    }
    # smp = 34560; quantize = Categorical(Rtn(60) + Gptq(180)) = 240
    # total = 34560 * 240 = 8294400
    assert PassSearchCommand._count_search_points(run_config) == 8294400


def test_count_search_points_with_alternative_pass_configs_params_excluded():
    # With include_pass_params defaulting to False, each config contributes a single point, so
    # the count reduces to the number of pass-config combinations: smp(1) * quantize(2) = 2.
    run_config = {
        "passes": {
            "smp": {"type": "SelectiveMixedPrecision"},
            "quantize": [{"type": "Rtn"}, {"type": "Gptq"}],
        },
        "engine": {"search_strategy": {"execution_order": "joint", "sampler": "sequential"}},
    }
    assert PassSearchCommand._count_search_points(run_config) == 2


def test_report_search_point_count_prints_count(capsys):
    run_config = {"passes": {"onnxconversion": {"type": "OnnxConversion"}}}
    command = PassSearchCommand.__new__(PassSearchCommand)
    command._report_search_point_count(run_config)
    out = capsys.readouterr().out
    assert "1 search point" in out


def test_report_search_point_count_prints_config_name_prefix(capsys):
    run_config = {
        "passes": {"mystery": {"type": "NotARealPassType"}},
        "engine": {"search_strategy": {"execution_order": "joint", "sampler": "sequential"}},
    }
    command = PassSearchCommand.__new__(PassSearchCommand)
    command._report_search_point_count(run_config, "config_00")
    out = capsys.readouterr().out
    assert "[config_00]" in out
    assert "Unable to determine" in out


# --------------------------------------------------------------------------- #
# Build-mode end-to-end (dry run / file generation)                           #
# --------------------------------------------------------------------------- #


def test_build_mode_dry_run_generates_config_files(tmp_path):
    output_dir = tmp_path / "search-out"
    command_args = [
        "pass-search",
        "-m",
        "microsoft/phi-2",
        "--passes",
        "OnnxConversion",
        "OnnxQuantization",
        "--dry_run",
        "-o",
        str(output_dir),
    ]

    cli_main(command_args)

    config_0 = output_dir / "config_00.json"
    config_1 = output_dir / "config_01.json"
    assert config_0.exists()
    assert config_1.exists()

    data_0 = json.loads(config_0.read_text())
    data_1 = json.loads(config_1.read_text())
    assert data_0["passes"] == {"onnxconversion": {"type": "OnnxConversion"}}
    assert data_1["passes"] == {"onnxquantization": {"type": "OnnxQuantization"}}
    # output_dir must use posix separators regardless of OS
    assert "\\" not in data_0["output_dir"]
    assert data_0["output_dir"].endswith("config_00")


def test_build_mode_dry_run_bracketed_group_single_config(tmp_path):
    output_dir = tmp_path / "search-out"
    command_args = [
        "pass-search",
        "-m",
        "microsoft/phi-2",
        "--passes",
        "[OnnxConversion, OnnxQuantization]",
        "--dry_run",
        "-o",
        str(output_dir),
    ]

    cli_main(command_args)

    config_0 = output_dir / "config_00.json"
    assert config_0.exists()
    assert not (output_dir / "config_01.json").exists()
    data = json.loads(config_0.read_text())
    assert list(data["passes"].keys()) == ["onnxconversion", "onnxquantization"]


@patch("olive.workflows.run")
def test_build_mode_runs_olive_for_each_group(mock_run, tmp_path):
    output_dir = tmp_path / "search-out"
    command_args = [
        "pass-search",
        "-m",
        "microsoft/phi-2",
        "--passes",
        "OnnxConversion",
        "OnnxQuantization",
        "-o",
        str(output_dir),
    ]

    cli_main(command_args)

    assert mock_run.call_count == 2


@patch("olive.cli.pass_search.PassSearchCommand._submit_remote_job", return_value="op_123")
def test_build_mode_submits_remote_job_when_key_provided(mock_submit, tmp_path):
    output_dir = tmp_path / "search-out"
    command_args = [
        "pass-search",
        "-m",
        "microsoft/phi-2",
        "--passes",
        "OnnxConversion",
        "--az_apim_subscription_key",
        "f" * 32,
        "-o",
        str(output_dir),
    ]

    cli_main(command_args)

    mock_submit.assert_called_once()


# --------------------------------------------------------------------------- #
# Config-file mode                                                            #
# --------------------------------------------------------------------------- #


@patch("olive.workflows.run")
def test_config_mode_runs_olive(mock_run, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"input_model": {"type": "HfModel"}, "output_dir": "out"}))
    command_args = ["pass-search", "--run-config", str(config_path)]

    cli_main(command_args)

    mock_run.assert_called_once()


@patch("olive.workflows.run")
def test_config_mode_applies_search_strategy(mock_run, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"input_model": {"type": "HfModel"}, "output_dir": "out"}))
    command_args = [
        "pass-search",
        "--run-config",
        str(config_path),
        "--sampler",
        "tpe",
        "--max-iter",
        "10",
    ]

    cli_main(command_args)

    passed_config = mock_run.call_args[0][0]
    assert passed_config["engine"]["search_strategy"] == {"sampler": "tpe", "max_iter": 10}


def test_config_mode_dry_run_does_not_run(tmp_path):
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    config_path.write_text(json.dumps({"input_model": {"type": "HfModel"}, "output_dir": str(output_dir)}))
    command_args = ["pass-search", "--run-config", str(config_path), "--dry_run"]

    with patch("olive.workflows.run") as mock_run:
        cli_main(command_args)

    mock_run.assert_not_called()


@patch("olive.cli.pass_search.PassSearchCommand._submit_remote_job", return_value="op_abc")
def test_config_mode_submits_remote_job_when_key_provided(mock_submit, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"input_model": {"type": "HfModel"}, "output_dir": "out"}))
    command_args = [
        "pass-search",
        "--run-config",
        str(config_path),
        "--az_apim_subscription_key",
        "1" * 32,
    ]

    with patch("olive.workflows.run") as mock_run:
        cli_main(command_args)

    mock_submit.assert_called_once()
    mock_run.assert_not_called()
