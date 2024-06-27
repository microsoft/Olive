# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.integ_test.utils import get_olive_workspace_config

import pytest

from olive.resource_path import ResourceType, create_resource_path

# pylint: disable=attribute-defined-outside-init


@pytest.mark.skip(reason="Skip AzureML related tests")
class TestAMLResourcePath:
    @pytest.fixture(autouse=True)
    def setup(self):
        workspace_config = get_olive_workspace_config()

        self.resource_path_configs = {
            ResourceType.AzureMLModel: {
                "type": ResourceType.AzureMLModel,
                "config": {"azureml_client": workspace_config, "name": "olive_model", "version": 1},
            },
            f"{ResourceType.AzureMLDatastore}_file": {
                "type": ResourceType.AzureMLDatastore,
                "config": {
                    "azureml_client": workspace_config,
                    "datastore_name": "workspaceblobstore",
                    "relative_path": "LocalUpload/91c260865404c817f0fb43e321137c32/model.onnx",
                },
            },
            f"{ResourceType.AzureMLDatastore}_folder": {
                "type": ResourceType.AzureMLDatastore,
                "config": {
                    "azureml_client": workspace_config,
                    "datastore_name": "workspaceblobstore",
                    "relative_path": "LocalUpload/db47b57f5cb1f9da097f8d6e540eaab0/local_folder/",
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
        [
            ResourceType.AzureMLModel,
            # TODO(trajep): remove skip once the bug of azureml-fsspec is fixed
            pytest.param(
                f"{ResourceType.AzureMLDatastore}_file",
                marks=pytest.mark.skip(reason="Credential bug in azureml-fsspec"),
            ),
            pytest.param(
                f"{ResourceType.AzureMLDatastore}_folder",
                marks=pytest.mark.skip(reason="Credential bug in azureml-fsspec"),
            ),
            ResourceType.AzureMLJobOutput,
        ],
    )
    def test_create_resource_path(self, resource_path_type):
        resource_path = create_resource_path(self.resource_path_configs[resource_path_type])
        assert "azureml" in resource_path.get_path()

    @pytest.mark.parametrize(
        "resource_path_type",
        [
            ResourceType.AzureMLModel,
            # TODO(trajep): remove skip once the bug of azureml-fsspec is fixed
            pytest.param(
                f"{ResourceType.AzureMLDatastore}_file",
                marks=pytest.mark.skip(reason="Credential bug in azureml-fsspec"),
            ),
            pytest.param(
                f"{ResourceType.AzureMLDatastore}_folder",
                marks=pytest.mark.skip(reason="Credential bug in azureml-fsspec"),
            ),
            ResourceType.AzureMLJobOutput,
        ],
    )
    def test_save_to_dir(self, resource_path_type, tmp_path):
        resource_path = create_resource_path(self.resource_path_configs[resource_path_type])

        save_result = "local_folder" if resource_path_type.endswith("folder") else "model.onnx"
        # test save to dir
        saved_resource = resource_path.save_to_dir(tmp_path / "save_to_dir")
        assert Path(saved_resource).exists()
        assert Path(saved_resource).name == save_result

        # test save to dir with new name
        new_name = "new_name"
        saved_resource = resource_path.save_to_dir(tmp_path / "save_to_dir", new_name)
        assert Path(saved_resource).exists()
        assert Path(saved_resource).stem == new_name

        # test fail to save to dir with existing name
        with pytest.raises(FileExistsError):
            resource_path.save_to_dir(tmp_path / "save_to_dir")

        # test save to dir with overwrite
        saved_resource = resource_path.save_to_dir(tmp_path / "save_to_dir", overwrite=True)
        assert Path(saved_resource).exists()
        assert Path(saved_resource).name == save_result
