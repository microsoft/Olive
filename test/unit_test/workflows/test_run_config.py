# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.common.pydantic_v1 import ValidationError
from olive.data.config import DataConfig
from olive.data.container.huggingface_container import HuggingfaceContainer
from olive.package_config import OlivePackageConfig
from olive.workflows.run.config import RunConfig
from olive.workflows.run.run import get_pass_module_path, is_execution_provider_required

# pylint: disable=attribute-defined-outside-init, unsubscriptable-object


class TestRunConfig:
    # like: Systems/Evaluation/Model and etc.
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.package_config = OlivePackageConfig.parse_file(OlivePackageConfig.get_default_config_path())

        # create a user_script.py file in the tmp_path and refer to it
        # this way the test can be run from any directory
        current_dir = Path(__file__).parent
        with open(current_dir / "mock_data" / "user_script.json") as f:
            user_script_json = json.load(f)
        user_script_json = json.dumps(user_script_json)

        user_script_py = tmp_path / "user_script.py"
        with open(user_script_py, "w") as f:
            f.write("")

        user_script_json = user_script_json.replace("user_script.py", user_script_py.as_posix())
        self.user_script_config_file = tmp_path / "user_script.json"
        with open(self.user_script_config_file, "w") as f:
            f.write(user_script_json)

    def test_config_without_azureml_config(self):
        with self.user_script_config_file.open() as f:
            user_script_config = json.load(f)

        user_script_config.pop("azureml_client")
        with pytest.raises(ValueError) as e:  # noqa: PT011
            RunConfig.parse_obj(user_script_config)
        assert "azureml_client is required for AzureML System but not provided." in str(e.value)

    @pytest.fixture()
    def mock_aml_credentials(self):
        # we need to mock all the credentials because the default credential will get tokens from all of them
        self.mocked_env_credentials = patch("azure.identity._credentials.default.EnvironmentCredential").start()
        self.mocked_managed_identity_credentials = patch(
            "azure.identity._credentials.default.ManagedIdentityCredential"
        ).start()
        self.mocked_shared_token_cache_credentials = patch(
            "azure.identity._credentials.default.SharedTokenCacheCredential"
        ).start()
        self.mocked_azure_cli_credentials = patch("azure.identity._credentials.default.AzureCliCredential").start()
        self.mocked_azure_powershell_credentials = patch(
            "azure.identity._credentials.default.AzurePowerShellCredential"
        ).start()
        self.mocked_interactive_browser_credentials = patch(
            "azure.identity._credentials.default.InteractiveBrowserCredential"
        ).start()
        yield
        patch.stopall()

    @pytest.mark.usefixtures("mock_aml_credentials")
    @pytest.mark.parametrize(
        "default_auth_params",
        [
            (None, (1, 1, 1, 1, 1, 0)),
            (
                {"exclude_environment_credential": True, "exclude_managed_identity_credential": False},
                (0, 1, 1, 1, 1, 0),
            ),
            ({"exclude_environment_credential": True, "exclude_managed_identity_credential": True}, (0, 0, 1, 1, 1, 0)),
        ],
    )
    def test_config_with_azureml_default_auth_params(self, default_auth_params):
        """default_auth_params[0] is a dict of the parameters to be passed to DefaultAzureCredential.

        default_auth_params[1] is a tuple of the number of times each credential is called.
        the order is totally same with that in DefaultAzureCredential where the credentials
        are called sequentially until one of them succeeds:
            EnvironmentCredential -> ManagedIdentityCredential -> SharedTokenCacheCredential
            -> AzureCliCredential -> AzurePowerShellCredential -> InteractiveBrowserCredential
        https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python
        """
        with self.user_script_config_file.open() as f:
            user_script_config = json.load(f)

        user_script_config["azureml_client"]["default_auth_params"] = default_auth_params[0]
        config = RunConfig.parse_obj(user_script_config)
        config.azureml_client.create_client()
        assert (
            self.mocked_env_credentials.call_count,
            self.mocked_managed_identity_credentials.call_count,
            self.mocked_shared_token_cache_credentials.call_count,
            self.mocked_azure_cli_credentials.call_count,
            self.mocked_azure_powershell_credentials.call_count,
            self.mocked_interactive_browser_credentials.call_count,
        ) == default_auth_params[1]

    @patch("azure.identity.DefaultAzureCredential")
    @patch("azure.identity.InteractiveBrowserCredential")
    def test_config_with_failed_azureml_default_auth(self, mocked_interactive_login, mocked_default_azure_credential):
        mocked_default_azure_credential.side_effect = Exception("mock error")
        with self.user_script_config_file.open() as f:
            user_script_config = json.load(f)
        config = RunConfig.parse_obj(user_script_config)
        config.azureml_client.create_client()
        assert mocked_interactive_login.call_count == 1

    def test_readymade_system(self):
        readymade_config_file = Path(__file__).parent / "mock_data" / "readymade_system.json"
        with readymade_config_file.open() as f:
            user_script_config = json.load(f)

        cfg = RunConfig.parse_obj(user_script_config)
        assert cfg.engine.target.config.accelerators[0].device.lower() == "gpu"
        assert cfg.engine.target.config.accelerators[0].execution_providers == ["CUDAExecutionProvider"]

    def test_default_engine(self):
        default_engine_config_file = Path(__file__).parent / "mock_data" / "default_engine.json"
        run_config = RunConfig.parse_file(default_engine_config_file)
        assert run_config.evaluators is None
        assert run_config.engine.host is None
        assert run_config.engine.target is None

    def test_deprecated_engine_ep(self):
        with self.user_script_config_file.open() as f:
            user_script_config = json.load(f)

        user_script_config["execution_providers"] = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
        with pytest.raises(ValidationError) as e:
            _ = RunConfig.parse_obj(user_script_config)
        errors = e.value.errors()
        assert errors[0]["loc"] == ("engine", "execution_providers")

    @pytest.mark.parametrize(("pass_type", "is_onnx"), [("IncQuantization", True), ("LoRA", False)])
    def test_get_module_path(self, pass_type, is_onnx):
        pass_module = get_pass_module_path(pass_type, self.package_config)
        assert pass_module.startswith("olive.passes.onnx") == is_onnx

    @pytest.mark.parametrize(
        ("passes", "pass_flows", "is_onnx"),
        [
            (None, None, True),
            (
                {
                    "lora": {"type": "LoRA"},
                },
                None,
                False,
            ),
            (
                {
                    "lora": {"type": "LoRA"},
                    "quantization": {"type": "IncQuantization"},
                },
                None,
                True,
            ),
            (
                {
                    "lora": {"type": "LoRA"},
                    "quantization": {"type": "IncQuantization"},
                },
                [["lora"]],
                False,
            ),
        ],
    )
    def test_is_execution_provider_required(self, passes, pass_flows, is_onnx):
        with self.user_script_config_file.open() as f:
            user_script_config = json.load(f)

        if passes:
            user_script_config["passes"] = passes
        if pass_flows:
            user_script_config["pass_flows"] = pass_flows

        run_config = RunConfig.parse_obj(user_script_config)
        result = is_execution_provider_required(run_config, self.package_config)
        assert result == is_onnx


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
            "passes": {"tuning": {"type": "OrtPerfTuning"}},
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
        config_dict["passes"]["tuning"]["data_config"] = data_config_str

        run_config = RunConfig.parse_obj(config_dict)
        pass_data_config = run_config.passes["tuning"].config["data_config"]
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
                "tuning": {
                    "type": "IncQuantization",
                }
            },
            "evaluate_input_model": False,
        }

    @pytest.mark.parametrize(
        ("search_strategy", "disable_search", "approach", "is_valid"),
        [
            (None, None, None, True),
            (None, None, "SEARCHABLE_VALUES", False),
            (None, False, "SEARCHABLE_VALUES", False),
            (None, None, "dummy_approach", True),
            (None, True, "dummy_approach", True),
            (
                {"execution_order": "joint", "search_algorithm": "exhaustive"},
                None,
                "SEARCHABLE_VALUES",
                True,
            ),
        ],
    )
    def test_pass_config_(self, search_strategy, disable_search, approach, is_valid):
        config_dict = self.template.copy()
        config_dict["search_strategy"] = search_strategy
        config_dict["passes"]["tuning"]["disable_search"] = disable_search
        config_dict["passes"]["tuning"]["approach"] = approach
        if not is_valid:
            with pytest.raises(ValueError):  # noqa: PT011
                RunConfig.parse_obj(config_dict)
        else:
            config = RunConfig.parse_obj(config_dict)
            assert config.passes["tuning"].config["approach"] == approach
