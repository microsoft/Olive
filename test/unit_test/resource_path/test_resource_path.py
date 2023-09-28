# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path

import pytest

from olive.resource_path import ResourceType, create_resource_path


class TestResourcePath:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = Path(self.tmp_dir.name).resolve()

        # create a local file
        self.local_file = self.tmp_dir_path / "local_file.txt"
        with self.local_file.open("w") as f:
            f.write("local file content")

        # create a local folder
        self.local_folder = self.tmp_dir_path / "local_folder"
        self.local_folder.mkdir()

        self.resource_path_configs = {
            ResourceType.LocalFile: self.local_file,
            ResourceType.LocalFolder: self.local_folder,
            ResourceType.StringName: "string_name",
        }

    @pytest.mark.parametrize(
        "resource_path_type",
        [ResourceType.LocalFile, ResourceType.LocalFolder, ResourceType.StringName],
    )
    def test_create_resource_path(self, resource_path_type):
        resource_path = create_resource_path(self.resource_path_configs[resource_path_type])
        assert resource_path.type == resource_path_type
        assert resource_path.get_path() == str(self.resource_path_configs[resource_path_type])

    @pytest.mark.parametrize(
        "resource_path_type",
        [ResourceType.LocalFile, ResourceType.LocalFolder],
    )
    def test_save_to_dir(self, resource_path_type):
        resource_path = create_resource_path(self.resource_path_configs[resource_path_type])

        # test save to dir
        saved_resource = resource_path.save_to_dir(self.tmp_dir_path / "save_to_dir")
        assert Path(saved_resource).exists()
        assert Path(saved_resource).name == Path(self.resource_path_configs[resource_path_type]).name

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
        assert Path(saved_resource).name == Path(self.resource_path_configs[resource_path_type]).name
