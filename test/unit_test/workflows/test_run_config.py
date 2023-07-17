# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.workflows.run.config import RunConfig


class TestRunConfig:
    # TODO: add more tests for different config files to test olive features
    # like: Systems/Evaluation/Model and etc.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.user_script_config_file = Path(__file__).parent / "mock_data" / "user_script.json"

    @pytest.mark.parametrize(
        "config_file",
        [
            Path(__file__).parent / "mock_data" / "transformer_dataset.json",
            Path(__file__).parent / "mock_data" / "only_transformer_dataset.json",
            Path(__file__).parent / "mock_data" / "ner_task_dataset.json",
        ],
    )
    def test_dataset_config_file(self, config_file):
        run_config = RunConfig.parse_file(config_file)
        for dc in run_config.data_configs.values():
            dc.to_data_container().create_dataloader()

    @pytest.mark.parametrize("system", ["local_system", "azureml_system"])
    def test_user_script_config(self, system):
        with open(self.user_script_config_file, "r") as f:
            user_script_config = json.load(f)

        user_script_config["engine"]["host"] = system
        user_script_config["engine"]["target"] = system
        config = RunConfig.parse_obj(user_script_config)
        for metric in config.evaluators["common_evaluator"].metrics:
            assert metric.user_config.data_dir.get_path().startswith("azureml://")

    def test_config_without_azureml_config(self):
        with open(self.user_script_config_file, "r") as f:
            user_script_config = json.load(f)

        user_script_config.pop("azureml_client")
        with pytest.raises(ValueError) as e:
            RunConfig.parse_obj(user_script_config)
            assert str(e.value) == "AzureML client config is required for AzureML system"

    @patch("azure.identity._credentials.default.InteractiveBrowserCredential")
    @patch("azure.identity._credentials.default.AzurePowerShellCredential")
    @patch("azure.identity._credentials.default.AzureCliCredential")
    @patch("azure.identity._credentials.default.SharedTokenCacheCredential")
    @patch("azure.identity._credentials.default.ManagedIdentityCredential")
    @patch("azure.identity._credentials.default.EnvironmentCredential")
    def test_config_with_azureml_default_auth_params(
        self,
        mocked_env_credential,
        mocked_managed_identity_credential,
        mocked_shared_token_cache_credential,
        mocked_azure_cli_credential,
        mocked_azure_powershell_credential,
        mocked_interactive_browser_credential,
    ):
        # we need to mock all the credentials because the default credential will get tokens from all of them
        with open(self.user_script_config_file, "r") as f:
            user_script_config = json.load(f)

        user_script_config["azureml_client"]["default_auth_params"] = None
        config = RunConfig.parse_obj(user_script_config)
        config.azureml_client.create_client()
        mocked_env_credential.assert_called_once()
        mocked_interactive_browser_credential.assert_not_called()

        user_script_config["azureml_client"]["default_auth_params"] = {
            "exclude_environment_credential": True,
            "exclude_managed_identity_credential": True,
        }
        config = RunConfig.parse_obj(user_script_config)
        config.azureml_client.create_client()
        assert mocked_env_credential.call_count == 1
        assert mocked_managed_identity_credential.call_count == 1
        assert mocked_azure_cli_credential.call_count == 2

        user_script_config["azureml_client"]["default_auth_params"] = {
            "exclude_environment_credential": True,
            "exclude_managed_identity_credential": False,
        }
        config = RunConfig.parse_obj(user_script_config)
        config.azureml_client.create_client()
        assert mocked_env_credential.call_count == 1
        assert mocked_managed_identity_credential.call_count == 2
        assert mocked_azure_cli_credential.call_count == 3

    def test_readymade_system(self):
        readymade_config_file = Path(__file__).parent / "mock_data" / "readymade_system.json"
        with open(readymade_config_file, "r") as f:
            user_script_config = json.load(f)

        cfg = RunConfig.parse_obj(user_script_config)
        assert cfg.engine.target.config.accelerators == ["GPU"]
