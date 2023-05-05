# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openvino.tools.pot import DataLoader

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.quantization import OpenVINOQuantization
from olive.snpe.utils.input_list import get_dir_members
from olive.workflows.wssi.config import ConvertQuantizeConfig
from olive.workflows.wssi.utils import get_model, resolve_model_dir

logger = logging.getLogger(__name__)


def openvino_convertquantize(config: ConvertQuantizeConfig):
    model = get_model(config.model)

    models_dir, name = resolve_model_dir(config.model, config.output_dir, config.output_name)

    # ------------------------------------------------------------------
    # OpenVINO model
    logger.info("Converting model to OpenVINO...")
    openvino_model_path = str(models_dir / f"{name}_openvino")

    convert_options = config.convert_options or {}
    convert_options.update(create_io_args(config.io_config.dict()))
    openvino_conversion = create_pass_from_dict(
        OpenVINOConversion, {"extra_config": convert_options}, disable_search=True
    )
    openvino_model = openvino_conversion.run(model, openvino_model_path)
    assert Path(openvino_model.model_path).is_dir()

    # ------------------------------------------------------------------
    # OpenVINO quantized model
    logger.info("Quantizing OpenVINO model...")
    openvino_quantized_model_path = str(models_dir / f"{name}_openvino_quantized")
    dataloader_func = lambda data_dir, batch_size: OVRawDataLoader(  # noqa: E731
        data_dir, config.io_config.input_names, config.io_config.input_shapes
    )

    quantize_options = config.quantize_options or {}
    openvino_quantization = create_pass_from_dict(
        OpenVINOQuantization,
        {"data_dir": config.quant_data, "dataloader_func": dataloader_func, **quantize_options},
        disable_search=True,
    )
    openvino_quantized_model = openvino_quantization.run(openvino_model, openvino_quantized_model_path)
    assert Path(openvino_quantized_model.model_path).is_dir()


def create_io_args(config: Dict[str, Any]) -> Dict[str, str]:
    io_args = {}
    if config["input_names"]:
        io_args["input"] = ",".join(config["input_names"])
    if config["input_shapes"]:
        assert len(config["input_shapes"]) == len(
            config["input_names"]
        ), "input_shapes and input_names must have the same length"
        io_args["input_shape"] = ",".join([str(x) for x in config["input_shapes"]])
        io_args["input_shape"] = io_args["input_shape"].replace(" ", "")
    if config["output_names"]:
        io_args["output"] = ",".join(config["output_names"])
    return io_args


class OVRawDataLoader(DataLoader):
    """
    OpenVINO Raw Data Loader
    """

    def __init__(self, data_dir: str, input_names: List[str], input_shapes: List[List[int]]):
        """
        data_dir: Directory containing the raw data files. If there is only one input, the data files should be in this
            directory. If there are multiple inputs, the data files should be in subdirectories named after the input
            names. The input is assumed to a binary file containing a float32 array.
        input_names: List of input names
        input_shapes: List of input shapes
        """
        self.data_dir = Path(data_dir).resolve()
        self.input_names = input_names
        assert len(input_names) == len(input_shapes), "Number of input names and shapes must match"
        self.input_shapes = input_shapes

        self.input_dirs = self.get_input_dirs(self.data_dir, self.input_names)
        self.input_members = self.get_input_members(input_names, self.input_dirs)

    def get_input_dirs(self, data_dir: Path, input_names: List[str]) -> Dict[str, Path]:
        """
        Get the input directories.
        If there is only one input, the data files should be in the data directory.
        If there are multiple inputs, the data files should be in subdirectories named after the input names.
        """
        input_dirs = {}
        if len(input_names) == 1:
            input_dirs[input_names[0]] = data_dir
        else:
            for input_name in input_names:
                input_dirs[input_name] = data_dir / input_name
        return input_dirs

    def get_input_members(self, input_names: List[str], input_dirs: Dict[str, Path]) -> List[Dict[str, Path]]:
        """
        Get the input members. All inputs must have the same number of members with the same names.
        Returns a list of dictionaries, where each dictionary contains the path to the data file for each input.
        """
        input_members = []
        members = get_dir_members(input_dirs[input_names[0]])
        for member in members:
            input_member = {}
            for input_name in input_names:
                input_member[input_name] = input_dirs[input_name] / member
            input_members.append(input_member)
        return input_members

    def __len__(self):
        return len(self.input_members)

    def __getitem__(self, index: int):
        input_member = self.input_members[index]
        data = {}
        for input_name in self.input_names:
            data[input_name] = self.load_input(input_name, input_member[input_name])
        return data, None

    def load_input(self, input_name: str, input_path: Path):
        """
        Load the input data from the input path.
        """
        input_shape = self.input_shapes[self.input_names.index(input_name)]
        input_data = np.fromfile(input_path, dtype=np.float32)
        input_data = input_data.reshape(input_shape)
        return input_data
