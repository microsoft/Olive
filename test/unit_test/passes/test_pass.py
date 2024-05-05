# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock

from olive.model.handler.onnx import ONNXModelHandler
from olive.passes.olive_pass import Pass


def check_filenames(model, model_path_dir, expected_files):
    # Check that the model has the additional_files attribute
    # and that it contains the expected files and that the files exist
    assert "additional_files" in model.model_attributes

    additional_files = model.model_attributes["additional_files"]
    assert len(additional_files) == len(expected_files)
    for file in expected_files:
        assert file in additional_files
        assert (model_path_dir / file).exists()


def test_pass_model_attributes_additional_files(tmp_path):
    model_1_path = tmp_path / "model_1"
    model_1_path.mkdir()

    model_2_path = tmp_path / "model_2"
    model_2_path.mkdir()
    model_2_filepath_1 = model_2_path / "model_2_file_1.txt"
    with open(model_2_filepath_1, "w") as strm:
        pass
    model_2_filepath_2 = model_2_path / "model_2_file_2.txt"
    with open(model_2_filepath_2, "w") as strm:
        strm.write("model_2_filepath_2")

    model_3_path = tmp_path / "model_3"
    model_3_path.mkdir()
    model_3_filepath_1 = model_3_path / "model_3_file_1.txt"
    with open(model_3_filepath_1, "w") as strm:
        pass
    model_3_filepath_2 = model_3_path / "model_3_file_2.txt"
    with open(model_3_filepath_2, "w") as strm:
        pass
    model_3_filepath_3 = model_3_path / "model_2_file_2.txt"
    with open(model_3_filepath_3, "w") as strm:
        strm.write("model_3_filepath_3")

    model_4_path = tmp_path / "model_4" / "model.onnx"
    model_4_path.parent.mkdir()
    with open(model_4_path, "w") as strm:
        pass

    def model_1_side_effect(arg):
        return str(model_1_path) if arg == "model_path" else None

    model_1 = MagicMock()
    model_1.get_resource = MagicMock(side_effect=model_1_side_effect)
    model_1.model_attributes = {}

    def model_2_side_effect(arg):
        return str(model_2_path) if arg == "model_path" else None

    model_2 = MagicMock()
    model_2.get_resource = MagicMock(side_effect=model_2_side_effect)
    model_2.model_attributes = {"additional_files": [model_2_filepath_1.name, model_2_filepath_2.name]}

    # Input model with no additional files, and output model with additional files
    Pass._carry_forward_additional_files(model_1, model_2)  # pylint: disable=W0212

    check_filenames(model_2, model_2_path, [model_2_filepath_1.name, model_2_filepath_2.name])

    def model_3_side_effect(arg):
        return str(model_3_path) if arg == "model_path" else None

    model_3 = MagicMock()
    model_3.get_resource = MagicMock(side_effect=model_3_side_effect)
    model_3.model_attributes = {
        "additional_files": [model_3_filepath_1.name, model_3_filepath_2.name, model_3_filepath_3.name]
    }

    # Both input & output models with additional files
    Pass._carry_forward_additional_files(model_2, model_3)  # pylint: disable=W0212

    # Input model should be unchanged
    check_filenames(model_2, model_2_path, [model_2_filepath_1.name, model_2_filepath_2.name])

    # Output model includes accumulated list of additional files
    check_filenames(
        model_3,
        model_3_path,
        [model_2_filepath_1.name, model_3_filepath_1.name, model_3_filepath_2.name, model_3_filepath_3.name],
    )

    # Pass 3 generated file shouldn't be overwritten by Pass 2 even though the file names are the same
    with open(model_3_filepath_3) as strm:
        content = strm.read()
    assert content == "model_3_filepath_3"

    def model_4_side_effect(arg):
        return str(model_4_path) if arg == "model_path" else None

    model_4 = MagicMock(spec=ONNXModelHandler)
    model_4.get_resource = MagicMock(side_effect=model_4_side_effect)
    model_4_set_resource_mock = MagicMock()
    model_4.set_resource = model_4_set_resource_mock
    model_4.model_attributes = {}

    # Output model's output path is a file rather than directory
    Pass._carry_forward_additional_files(model_3, model_4)  # pylint: disable=W0212

    # Input model should be unchanged
    check_filenames(
        model_3,
        model_3_path,
        [model_2_filepath_1.name, model_3_filepath_1.name, model_3_filepath_2.name, model_3_filepath_3.name],
    )

    # check that the model_path resource was changed to the parent directory
    # for some reason called_once_with doesn't work here for Path objects
    assert model_4_set_resource_mock.call_count == 1
    assert model_4_set_resource_mock.call_args_list[0][0] == ("model_path", model_4_path.parent)

    # All files from input should be carried forward to output models parent directory
    check_filenames(
        model_4,
        model_4_path.parent,
        [model_2_filepath_1.name, model_2_filepath_2.name, model_3_filepath_1.name, model_3_filepath_2.name],
    )
