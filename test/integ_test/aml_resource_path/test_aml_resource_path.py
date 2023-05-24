# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import tempfile
from pathlib import Path

import pytest

from olive.logging import set_default_logger_severity
from olive.resource_path import ResourceType, create_resource_path

set_default_logger_severity(0)


class TestAMLResourcePath:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = Path(self.tmp_dir.name).resolve()

        workspace_config = get_olive_workspace_config()

        self.resource_path_configs = {
            ResourceType.AzureMLModel: {
                "type": ResourceType.AzureMLModel,
                "config": {"azureml_client": workspace_config, "name": "olive_model", "version": 1},
            },
            ResourceType.AzureMLDatastore: {
                "type": ResourceType.AzureMLDatastore,
                "config": {
                    "azureml_client": workspace_config,
                    "datastore_name": "workspaceblobstore",
                    "relative_path": "LocalUpload/91c260865404c817f0fb43e321137c32/model.onnx",
                },
            },
            ResourceType.AzureMLJobOutput: {
                "type": ResourceType.AzureMLJobOutput,
                "config": {
                    "azureml_client": workspace_config,
                    "job_name": "23add417-6f33-4d4b-85aa-3f33d2b669a8",
                    "output_name": "pipeline_output",
                    "relative_path": "model.onnx",
                },
            },
        }

    @pytest.mark.parametrize(
        "resource_path_type",
        [ResourceType.AzureMLModel, ResourceType.AzureMLDatastore, ResourceType.AzureMLJobOutput],
    )
    def test_create_resource_path(self, resource_path_type):
        resource_path = create_resource_path(self.resource_path_configs[resource_path_type])
        assert resource_path.type == resource_path_type
        assert "azureml" in resource_path.get_path()

    @pytest.mark.parametrize(
        "resource_path_type",
        [ResourceType.AzureMLModel, ResourceType.AzureMLDatastore, ResourceType.AzureMLJobOutput],
    )
    def test_save_to_dir(self, resource_path_type):
        resource_path = create_resource_path(self.resource_path_configs[resource_path_type])

        # test save to dir
        saved_resource = resource_path.save_to_dir(self.tmp_dir_path / "save_to_dir")
        assert Path(saved_resource).exists()
        assert Path(saved_resource).name == "model.onnx"

        # test save to dir with new name
        saved_resource = resource_path.save_to_dir(self.tmp_dir_path / "save_to_dir", "new_name")
        assert Path(saved_resource).exists()
        assert Path(saved_resource).stem == "new_name"

        # test fail to save to dir with existing name
        with pytest.raises(FileExistsError):
            resource_path.save_to_dir(self.tmp_dir_path / "save_to_dir")

        # test save to dir with overwrite
        saved_resource = resource_path.save_to_dir(self.tmp_dir_path / "save_to_dir", overwrite=True)
        assert Path(saved_resource).exists()
        assert Path(saved_resource).name == "model.onnx"


def get_olive_workspace_config():
    subscription_id = os.environ.get("WORKSPACE_SUBSCRIPTION_ID")
    if subscription_id is None:
        raise Exception("Please set the environment variable WORKSPACE_SUBSCRIPTION_ID")

    resource_group = os.environ.get("WORKSPACE_RESOURCE_GROUP")
    if resource_group is None:
        raise Exception("Please set the environment variable WORKSPACE_RESOURCE_GROUP")

    workspace_name = os.environ.get("WORKSPACE_NAME")
    if workspace_name is None:
        raise Exception("Please set the environment variable WORKSPACE_NAME")

    workspace_config = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
    }

    return workspace_config
