# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import List, Union


def resolve_input_list(
    data_dir: str,
    input_list_file: str,
    dest_dir: str,
    input_path_parent: str = None,
    resolved_filename: str = "input_list.txt",
) -> str:
    """Resolve the paths in an input list file to absolute paths.

    data_dir: The directory to make the resolved paths children of.
    input_list_file: The input list file to resolve. Unless input_path_parent is specified, the paths in the input
        list file are assumed to be relative to data_dir.
    dest_dir: The directory to write the resolved input list file to.
    input_path_parent: The directory paths in the input list file, if absolute, are children of.
        This is used for android targets where the input list file is already resolved to absolute paths and we need
        create a new input list file for the target data directory.
    resolved_filename: The name of the resolved input list file.
    """
    resolved_input_list_file = (Path(dest_dir) / resolved_filename).resolve()

    with Path(input_list_file).open() as input_list, resolved_input_list_file.open("w") as resolved_input_list:
        for line in input_list:
            # skip output lines
            if line.startswith(("#", "%")):
                resolved_input_list.write(line)
                continue

            # split the line into inputs
            inputs = line.strip().split(" ")
            if len(inputs) > 1 and not all(":=" in x for x in inputs):
                raise ValueError(
                    "Invalid input list. For multiple inputs, input lines must be of the form:"
                    " <input_layer_name>:=<input_layer_path>[<space><input_layer_name>:=<input_layer_path>]"
                )
            # line to write to the resolved input list
            inputs_line = ""
            for input_item in inputs:
                if ":=" in input_item:
                    # multiple inputs on the same line
                    input_name, input_path = input_item.split(":=")
                    if input_path_parent is not None:
                        input_path = Path(input_path).relative_to(Path(input_path_parent))
                    input_path = (Path(data_dir) / input_path).as_posix()
                    inputs_line += f"{input_name}:={input_path} "
                else:
                    input_path = input_item
                    if input_path_parent is not None:
                        input_path = Path(input_path).relative_to(Path(input_path_parent))
                    input_path = (Path(data_dir) / input_path).as_posix()
                    inputs_line += f"{input_path} "
            resolved_input_list.write(inputs_line.strip() + "\n")

    return str(resolved_input_list_file)


def get_dir_members(dir_path: str) -> set:
    """Get the members of a directory in sorted order."""
    members = [member.relative_to(dir_path) for member in dir_path.glob("**/*")]
    return sorted(set(members))


def create_input_list(
    data_dir: str,
    input_names: List[str],
    input_dirs: List[Union[str, None]] = None,
    input_list_file: str = None,
    add_input_names: bool = False,
    add_output_names: bool = False,
    output_names: List[str] = None,
    append_0: bool = False,
    num_samples: int = None,
) -> str:
    """Create an input list file for a data directory.

    data_dir: The data directory to create the input list file for.
    input_names: The names of the inputs.
    input_dirs: The sub-directories of data_dir containing the input data.
        If not specified, the input directories are assumed to be the same as the input names.
        List values can be None if the input directory is the same as the input name.
        We assume all input directories have the same members.
    input_list_file: The path to the input list file to create. If not specified, the input list file is created in
        the data directory with the name 'input_list.txt'.
    add_input_names: If True, the input names are added to the input list file. Only used if there is one input, for
        multiple inputs, the input names are always added.
    add_output_names: If True, the output names are added to the input list file.
    output_names: The names of the outputs. Required if add_output_names is True.
    append_0: If True, a ":0" is appended to the input names. For some reason, SNPE attaches a ":0" to the input and
        output names when converting TensorFlow models to SNPE DLC.
        TODO[jiapli]: To be verify for QNN SDK.
    num_samples: The number of samples to add to the input list file. If not specified, all samples are added.
    """
    data_dir_path = Path(data_dir).resolve()
    if not data_dir_path.is_dir():
        raise FileNotFoundError(f"Data directory '{data_dir}' ({str(data_dir_path)}) does not exist")

    # path to data directory to each input relative to data_dir
    input_dirs = input_dirs or input_names
    # resolve None values in input_dirs to corresponding input_names
    input_dirs = [input_dir or input_name for input_dir, input_name in zip(input_dirs, input_names)]
    for input_dir in input_dirs:
        if not (data_dir_path / input_dir).is_dir():
            raise FileNotFoundError(
                f"Input directory '{input_dir}' does not exist in data directory '{data_dir}' ({data_dir_path})"
            )

    # add input names to input_list
    # always add input names if there is more than one input
    add_input_names = add_input_names or len(input_names) > 1

    input_dir_members = get_dir_members(data_dir_path / input_dirs[0])
    # check all input dirs have the same member file names
    if len(input_names) > 1:
        for input_dir in input_dirs:
            members = get_dir_members(data_dir_path / input_dir)
            if members != input_dir_members:
                raise ValueError(f"Input directories {input_dirs[0]}' and '{input_dir}' do not have the same members")

    input_list_content = ""
    zero_str = ":0" if append_0 else ""
    if add_output_names:
        if not output_names:
            raise ValueError("Output names must be specified if add_output_names is True")
        input_list_content += "%" + " ".join([f"{output_name}{zero_str}" for output_name in output_names]) + "\n"
    if num_samples is not None:
        input_dir_members = input_dir_members[:num_samples]
    for member in input_dir_members:
        if not add_input_names:
            input_list_content += f"{input_dirs[0]}/{member.as_posix()}" + "\n"
        else:
            input_list_content += (
                " ".join(
                    [
                        f"{input_name}{zero_str}:={input_dir}/{member.as_posix()}"
                        for input_name, input_dir in zip(input_names, input_dirs)
                    ]
                )
                + "\n"
            )

    input_list_file = input_list_file or str(data_dir_path / "input_list.txt")
    with Path(input_list_file).open("wb") as f:
        f.write(input_list_content.encode("utf-8"))

    return input_list_file


def get_input_list(data_dir: str, input_list_file: str, tmp_dir: str):
    """Get the resolved input list file.

    data_dir: The data directory.
    input_list_file: Name of the input list file. This file is assumed to be in the data directory.
    tmp_dir: The directory to write the resolved input list file to.
    """
    data_dir_path = Path(data_dir).resolve()
    if not data_dir_path.is_dir():
        raise FileNotFoundError(f"Data directory '{data_dir}' ({str(data_dir_path)}) does not exist")

    input_list_file_path = data_dir_path / input_list_file
    if not input_list_file_path.is_file():
        raise FileNotFoundError(
            f"Input list '{input_list_file}' does not exist in data directory '{data_dir}' ({str(data_dir_path)})"
        )

    return resolve_input_list(data_dir, str(input_list_file_path), tmp_dir)


def get_input_ids(input_list_file: str) -> List[str]:
    """Get the input IDs from an input list file.

    Only returns one id per sample (line) if there are multiple inputs.
    Assumes all inputs have the same id for a given sample.
    """
    input_ids = []
    with Path(input_list_file).open() as input_list:
        for line in input_list:
            # skip output lines
            if line.startswith(("#", "%")):
                continue

            # split the line into inputs
            inputs = line.strip().split(" ")
            # get the id from the first input
            input_id = Path(inputs[0].split(":=")[-1]).stem
            input_ids.append(input_id)

    return input_ids
