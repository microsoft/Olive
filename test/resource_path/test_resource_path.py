# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from pathlib import Path

import pytest

from olive.resource_path import ResourceType, create_resource_path, find_all_resources

# pylint: disable=attribute-defined-outside-init


class TestResourcePath:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_dir_path = tmp_path

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

        self.all_resources_configs = [
            ([], None, {}),
            ({}, None, {}),
            ("string_name", None, {}),
            ([self.local_file], None, {(0,): self.local_file}),
            (
                [self.local_file, self.local_folder, "string_name"],
                None,
                {(0,): self.local_file, (1,): self.local_folder},
            ),
            ({"key0": self.local_file}, None, {("key0",): self.local_file}),
            (
                {"key0": self.local_file, "key1": {"key2": self.local_folder}, "key3": "string_name"},
                None,
                {("key0",): self.local_file, ("key1", "key2"): self.local_folder},
            ),
            (
                {"key0": self.local_file, "key1": [self.local_folder]},
                None,
                {("key0",): self.local_file, ("key1", 0): self.local_folder},
            ),
            (
                {"key0": self.local_file, "key1": {"key2": self.local_folder}, "key3": "string_name"},
                ["key2"],
                {("key0",): self.local_file},
            ),
            # this case is to ensure the `NestedConfig` config gathering doesn't modify the original config
            (
                {"key0": "string_name", "config": {"key1": self.local_folder}, "key2": "string_name"},
                None,
                {("config", "key1"): self.local_folder},
            ),
        ]

    @pytest.mark.parametrize(
        "resource_path_type",
        [ResourceType.LocalFile, ResourceType.LocalFolder, ResourceType.StringName],
    )
    def test_create_resource_path(self, resource_path_type):
        resource_path = create_resource_path(self.resource_path_configs[resource_path_type])
        assert resource_path.type == resource_path_type
        assert resource_path.get_path() == str(self.resource_path_configs[resource_path_type])

    @pytest.mark.parametrize("value", [None, 1, object()])
    def test_resource_type_rejects_non_string_value(self, value):
        with pytest.raises(ValueError, match="is not a valid ResourceType"):
            ResourceType(value)

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

    def test_find_all_resources(self):
        for resources, ignore_keys, expected in self.all_resources_configs:
            original_resources = deepcopy(resources)
            for key, value in find_all_resources(resources, ignore_keys=ignore_keys).items():
                assert key in expected
                assert value.get_path() == str(expected[key])
            # ensure the original resources are not modified
            assert resources == original_resources

    @pytest.mark.parametrize("is_directory", [False, True])
    @pytest.mark.parametrize("explicit_overwrite", [False, True])
    def test_save_to_dir_rejects_existing_entries_when_flattening(self, is_directory, explicit_overwrite):
        source = self.local_folder / "model"
        output_dir = self.tmp_dir_path / "output"
        output_dir.mkdir()
        target = output_dir / source.name
        if is_directory:
            source.mkdir()
            target.mkdir()
            source = source / "weights.bin"
            target = target / "weights.bin"
        source.write_text("new weights")
        target.write_text("original weights")
        (self.local_folder / "new_config.json").write_text("new config")
        resource_path = create_resource_path(self.local_folder)

        kwargs = {"overwrite": False} if explicit_overwrite else {}
        with pytest.raises(FileExistsError, match="overwrite is set to False"):
            resource_path.save_to_dir(output_dir, flatten=True, **kwargs)

        assert target.read_text() == "original weights"
        assert source.read_text() == "new weights"
        assert not (output_dir / "new_config.json").exists()

    @pytest.mark.parametrize("is_directory", [False, True])
    def test_save_to_dir_replaces_existing_entries_when_flattening_with_overwrite(self, is_directory):
        source = self.local_folder / "model"
        output_dir = self.tmp_dir_path / "output"
        output_dir.mkdir()
        target = output_dir / source.name
        if is_directory:
            source.mkdir()
            target.mkdir()
            (target / "obsolete.bin").write_text("obsolete weights")
            source = source / "weights.bin"
            target = target / "weights.bin"
        source.write_text("new weights")
        target.write_text("original weights")
        (output_dir / "unrelated.txt").write_text("keep me")
        resource_path = create_resource_path(self.local_folder)

        resource_path.save_to_dir(output_dir, flatten=True, overwrite=True)

        assert target.read_text() == "new weights"
        assert source.read_text() == "new weights"
        assert not (output_dir / "model" / "obsolete.bin").exists()
        assert (output_dir / "unrelated.txt").read_text() == "keep me"

    @pytest.mark.parametrize("overwrite", [False, True])
    def test_save_to_dir_preserves_unrelated_entries_when_flattening(self, overwrite):
        (self.local_folder / "model.onnx").write_text("model")
        output_dir = self.tmp_dir_path / "output"
        output_dir.mkdir()
        (output_dir / "unrelated.txt").write_text("keep me")
        resource_path = create_resource_path(self.local_folder)

        resource_path.save_to_dir(output_dir, flatten=True, overwrite=overwrite)

        assert (output_dir / "model.onnx").read_text() == "model"
        assert (output_dir / "unrelated.txt").read_text() == "keep me"
