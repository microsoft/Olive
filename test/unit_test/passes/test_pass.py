# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from unittest.mock import MagicMock, patch

from olive.model.handler.onnx import ONNXModelHandler
from olive.passes.olive_pass import Pass


@patch("shutil.copy")
def test_pass_model_attributes_additional_files(self, tmpdir):
    tmpdir = Path(tmpdir)

    model_1_path = tmpdir / "model_1"
    model_1_path.mkdir()

    model_2_path = tmpdir / "model_2"
    model_2_path.mkdir()
    model_2_filepath_1 = model_2_path / "model_2_file_1.txt"
    with open(model_2_filepath_1, "w") as strm:
        pass
    model_2_filepath_2 = model_2_path / "model_2_file_2.txt"
    with open(model_2_filepath_2, "w") as strm:
        strm.write("model_2_filepath_2")

    model_3_path = tmpdir / "model_3"
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

    model_4_path = tmpdir / "model_4" / "model.onnx"
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
    model_2.model_attributes = {"additional_files": [str(model_2_filepath_1), str(model_2_filepath_2)]}

    # Input model with no additional files, and output model with additional files
    Pass._carry_forward_additional_files(model_1, model_2)  # pylint: disable=W0212

    assert "additional_files" in model_2.model_attributes
    assert len(model_2.model_attributes["additional_files"]) == 2
    assert str(model_2_filepath_1) in model_2.model_attributes["additional_files"]
    assert str(model_2_filepath_2) in model_2.model_attributes["additional_files"]

    def model_3_side_effect(arg):
        return str(model_3_path) if arg == "model_path" else None

    model_3 = MagicMock()
    model_3.get_resource = MagicMock(side_effect=model_3_side_effect)
    model_3.model_attributes = {
        "additional_files": [str(model_3_filepath_1), str(model_3_filepath_2), str(model_3_filepath_3)]
    }

    # Both input & output models with additional files
    Pass._carry_forward_additional_files(model_2, model_3)  # pylint: disable=W0212

    # Input model should be unchanged
    assert "additional_files" in model_2.model_attributes
    assert len(model_2.model_attributes["additional_files"]) == 2
    assert str(model_2_filepath_1) in model_2.model_attributes["additional_files"]
    assert str(model_2_filepath_2) in model_2.model_attributes["additional_files"]

    # Output model includes accumulated list of additional files
    assert "additional_files" in model_3.model_attributes
    assert len(model_3.model_attributes["additional_files"]) == 4
    assert str(model_3_path / "model_2_file_1.txt") in model_3.model_attributes["additional_files"]
    assert str(model_3_path / "model_2_file_2.txt") in model_3.model_attributes["additional_files"]
    assert str(model_3_filepath_1) in model_3.model_attributes["additional_files"]
    assert str(model_3_filepath_2) in model_3.model_attributes["additional_files"]

    # Pass 3 generated file shouldn't be overwritten by Pass 2 even though the file names are the same
    with open(str(model_3_filepath_3)) as strm:
        content = strm.read()
    assert content == "model_3_filepath_3"

    model_4 = ONNXModelHandler(model_path=str(model_4_path))

    # Output model's output path is a file rather than directory
    Pass._carry_forward_additional_files(model_3, model_4)  # pylint: disable=W0212

    # Input model should be unchanged
    assert "additional_files" in model_3.model_attributes
    assert len(model_3.model_attributes["additional_files"]) == 4
    assert str(model_3_path / "model_2_file_1.txt") in model_3.model_attributes["additional_files"]
    assert str(model_3_path / "model_2_file_2.txt") in model_3.model_attributes["additional_files"]
    assert str(model_3_filepath_1) in model_3.model_attributes["additional_files"]
    assert str(model_3_filepath_2) in model_3.model_attributes["additional_files"]

    # All files from input should be carried forward to output models parent directory
    assert "additional_files" in model_4.model_attributes
    assert len(model_4.model_attributes["additional_files"]) == 4
    assert str(model_4_path.parent / "model_2_file_1.txt") in model_4.model_attributes["additional_files"]
    assert str(model_4_path.parent / "model_2_file_2.txt") in model_4.model_attributes["additional_files"]
    assert str(model_4_path.parent / "model_3_file_1.txt") in model_4.model_attributes["additional_files"]
    assert str(model_4_path.parent / "model_3_file_2.txt") in model_4.model_attributes["additional_files"]
