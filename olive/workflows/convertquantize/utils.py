# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np

from olive.model import OliveModel, ONNXModel, TensorFlowModel
from olive.snpe.utils.input_list import create_input_list

logger = logging.getLogger(__name__)


def get_model(model_path: Path) -> OliveModel:
    if model_path.suffix == ".onnx":
        logger.info("Loading model...")
        model = ONNXModel(model_path)
    elif model_path.suffix == ".pb":
        logger.info("Loading model...")
        model = TensorFlowModel(model_path, is_file=Path(model_path).is_file())
    else:
        raise Exception(f"Unsupported model format: {model_path.suffix}")
    return model


def resolve_model_dir(model_path: Path, output_dir: Path = None, output_name: str = None) -> Tuple[Path, str]:
    models_dir = model_path.resolve().parent if output_dir is None else output_dir.resolve()
    name = model_path.resolve().stem if output_name is None else output_name
    return models_dir, name


def prepare_snpe_quant_data(data_dir: Path, source_io_config: dict, target_io_config: dict, workspace: Path) -> str:
    data_dir = data_dir.resolve()
    num_inputs = len(source_io_config["input_names"])
    for idx in range(num_inputs):
        # input name
        input_name = source_io_config["input_names"][idx]
        logger.info(f"Processing input '{input_name}'...")

        # shapes of source model and snpe model
        source_shape = source_io_config["input_shapes"][idx]
        target_shape = target_io_config["input_shapes"][idx]

        # subdirectory of data_dir for this input
        input_dir = data_dir / input_name if num_inputs > 1 else data_dir
        new_input_dir = workspace / input_name
        new_input_dir.mkdir(parents=True, exist_ok=True)

        if source_shape == target_shape:
            logger.info("Source and target shapes are the same, copying data...")
            shutil.copytree(input_dir, new_input_dir, dirs_exist_ok=True)
        else:
            logger.info("Source and target shapes are different, transposing data...")
            # find the permutation of the source shape that matches the target shape
            # e.g. source_shape = [1, 3, 224, 224], target_shape = [1, 224, 224, 3]
            #      -> permutation = [0, 2, 3, 1]
            # NCDHW -> NDHWC, NCHW -> NHWC, NFC -> NCF
            channel_permutation = [0] + list(range(2, len(source_shape))) + [1]
            # NTF -> TNF
            # TODO: confirm if it is NTF -> TNF or TNF -> NTF. Doesn't really matter since the first two dimensions are
            # transposed anyway
            time_permutation = [1, 0] + list(range(2, len(source_shape)))
            if target_shape == [source_shape[idx] for idx in channel_permutation]:
                permutation = channel_permutation
            elif target_shape == [source_shape[idx] for idx in time_permutation]:
                permutation = time_permutation
            else:
                error_message = (
                    f"Cannot find a valid permutation of the source shape {source_shape} that matches the target shape"
                    f" {target_shape}"
                )
                logger.error(error_message)
                raise Exception(error_message)
            logger.info(f"Source shape: {source_shape}, target shape: {target_shape}, permutation: {permutation}")

            for member in input_dir.iterdir():
                input_data = np.fromfile(member, dtype=np.float32)
                input_data = input_data.reshape(source_shape)
                input_data = np.transpose(input_data, permutation)
                input_data.tofile(new_input_dir / member.name)

    # create input list file
    logger.info("Creating input list file...")
    source_input_names = source_io_config["input_names"]
    target_input_names = target_io_config["input_names"]
    target_output_names = target_io_config["output_names"]
    input_list_file = create_input_list(
        data_dir=str(workspace),
        input_names=target_input_names,
        input_dirs=source_input_names,
        add_input_names=len(source_input_names) > 1,
        add_output_names=len(target_output_names) > 1,
        output_names=target_output_names,
    )

    return input_list_file
