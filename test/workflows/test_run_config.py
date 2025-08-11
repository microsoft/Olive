# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path

import pytest

from olive.common.pydantic_v1 import ValidationError
from olive.data.config import DataConfig
from olive.data.container.huggingface_container import HuggingfaceContainer
from olive.package_config import OlivePackageConfig
from olive.workflows.run.config import RunConfig

# pylint: disable=attribute-defined-outside-init, unsubscriptable-object


class TestRunConfig:
    # like: Systems/Evaluation/Model and etc.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.package_config = OlivePackageConfig.parse_file(OlivePackageConfig.get_default_config_path())

    def test_default_engine(self):
        default_engine_config_file = Path(__file__).parent / "mock_data" / "default_engine.json"
        run_config = RunConfig.parse_file(default_engine_config_file)
        assert run_config.evaluators is None
        assert run_config.engine.host is None
        assert run_config.engine.target is None

    @pytest.mark.parametrize(("pass_type", "is_onnx"), [("IncQuantization", True), ("LoRA", False)])
    def test_get_module_path(self, pass_type, is_onnx):
        assert self.package_config.is_onnx_module(pass_type) == is_onnx


class TestDataConfigValidation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.template = {
            "input_model": {
                "type": "HfModel",
                "model_path": "dummy_model",
                "task": "dummy_task",
            },
            "data_configs": [
                {
                    "name": "dummy_data_config2",
                    "type": HuggingfaceContainer.__name__,
                    "load_dataset_config": {"data_name": "dummy_dataset2"},
                    "pre_process_data_config": {
                        "model_name": "dummy_model2",
                        "task": "dummy_task2",
                    },
                }
            ],
            "passes": {"tuning": [{"type": "OrtSessionParamsTuning"}]},
            "engine": {"evaluate_input_model": False},
        }

    @pytest.mark.parametrize(
        ("model_name", "task", "expected_model_name", "expected_task"),
        [
            ("dummy_model2", "dummy_task2", "dummy_model2", "dummy_task2"),  # no auto insert
            ("dummy_model2", None, "dummy_model2", "dummy_task"),  # auto insert task
            (None, "dummy_task2", "dummy_model", "dummy_task2"),  # auto insert model_name
            (None, None, "dummy_model", "dummy_task"),  # auto insert model_name and task
        ],
    )
    def test_auto_insert_model_name_and_task(self, model_name, task, expected_model_name, expected_task):
        config_dict = deepcopy(self.template)
        config_dict["data_configs"] = [
            {
                "name": "dummy_data_config2",
                "type": HuggingfaceContainer.__name__,
                "load_dataset_config": {"data_name": "dummy_dataset2"},
                "pre_process_data_config": {"model_name": model_name, "task": task},
                "post_process_data_config": {"model_name": model_name, "task": task},
            }
        ]

        run_config = RunConfig.parse_obj(config_dict)
        assert run_config.data_configs[0].name == "dummy_data_config2"
        assert run_config.data_configs[0].pre_process_params.get("model_name") == expected_model_name
        assert run_config.data_configs[0].pre_process_params.get("task") == expected_task
        assert run_config.data_configs[0].post_process_params.get("model_name") == expected_model_name
        assert run_config.data_configs[0].post_process_params.get("task") == expected_task

    # works similarly for trust_remote_args
    @pytest.mark.parametrize(
        ("has_load_kwargs", "trust_remote_code", "data_config_trust_remote_code", "expected_trust_remote_code"),
        [
            (False, None, None, None),
            (False, None, True, True),
            (True, True, None, True),
            (True, None, None, None),
            (True, None, True, True),
            (True, None, False, False),
            (True, True, False, False),
            (True, False, True, True),
        ],
    )
    def test_auto_insert_trust_remote_code(
        self, has_load_kwargs, trust_remote_code, data_config_trust_remote_code, expected_trust_remote_code
    ):
        config_dict = deepcopy(self.template)
        if has_load_kwargs:
            config_dict["input_model"]["load_kwargs"] = {"trust_remote_code": trust_remote_code}
        if data_config_trust_remote_code is not None:
            config_dict["data_configs"] = [
                {
                    "name": "dummy_data_config2",
                    "type": HuggingfaceContainer.__name__,
                    "pre_process_data_config": {"trust_remote_code": data_config_trust_remote_code},
                    "load_dataset_config": {"trust_remote_code": data_config_trust_remote_code},
                }
            ]

        run_config = RunConfig.parse_obj(config_dict)

        assert run_config.data_configs[0].name == "dummy_data_config2"
        assert run_config.data_configs[0].load_dataset_params.get("trust_remote_code") == expected_trust_remote_code
        assert run_config.data_configs[0].pre_process_params.get("trust_remote_code") == expected_trust_remote_code

    @pytest.mark.parametrize(
        "data_config_str",
        [None, "dummy_data_config2"],
    )
    def test_str_to_data_config(self, data_config_str):
        config_dict = deepcopy(self.template)
        config_dict["passes"]["tuning"][0]["data_config"] = data_config_str

        run_config = RunConfig.parse_obj(config_dict)
        pass_data_config = run_config.passes["tuning"][0].config["data_config"]
        if data_config_str is None:
            assert pass_data_config is None
        else:
            assert isinstance(pass_data_config, DataConfig)
            assert pass_data_config.name == data_config_str


class TestPassConfigValidation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.template = {
            "input_model": {
                "type": "OnnxModel",
            },
            "passes": {
                "tuning": [
                    {
                        "type": "IncQuantization",
                    }
                ]
            },
            "evaluate_input_model": False,
        }

    @pytest.mark.parametrize(
        ("search_strategy", "approach", "is_valid"),
        [
            (None, None, True),
            (None, "SEARCHABLE_VALUES", False),
            (None, "dummy_approach", True),
            (
                {"execution_order": "joint", "sampler": "sequential"},
                "SEARCHABLE_VALUES",
                False,
            ),
        ],
    )
    def test_pass_config_(self, search_strategy, approach, is_valid):
        config_dict = self.template.copy()
        config_dict["search_strategy"] = search_strategy
        config_dict["passes"]["tuning"][0]["approach"] = approach
        if not is_valid:
            with pytest.raises(ValueError):  # noqa: PT011
                RunConfig.parse_obj(config_dict)
        else:
            config = RunConfig.parse_obj(config_dict)
            assert config.passes["tuning"][0].config["approach"] == approach


class TestPythonEnvironmentSystemConfig:
    @pytest.mark.parametrize(
        ("python_environment", "error_message"),
        [
            (
                {
                    "type": "PythonEnvironment",
                    "config": {
                        "python_environment_path": "invalid_path",
                    },
                },
                "Python path.*invalid_path does not exist",
            ),
            (
                {
                    "type": "PythonEnvironment",
                },
                "python_environment_path is required for PythonEnvironmentSystem native mode",
            ),
        ],
    )
    def test_python_environment_path(self, python_environment, error_message):
        invalid_config = {
            "input_model": {
                "type": "OnnxModel",
            },
            "passes": {
                "tuning": [
                    {
                        "type": "IncQuantization",
                    }
                ]
            },
            "systems": {"py_system": python_environment},
        }
        with pytest.raises(ValidationError, match=error_message):
            RunConfig.parse_obj(invalid_config)
