# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional, Union

import onnx

from olive.model.utils.onnx_utils import resolve_onnx_path

logger = logging.getLogger(__name__)


def model_proto_to_file(
    model: onnx.ModelProto,
    output_path: Union[str, Path],
    save_as_external_data: Optional[bool] = False,
    all_tensors_to_one_file: Optional[bool] = True,
    external_data_name: Optional[Union[str, Path]] = None,
    size_threshold: Optional[int] = 1024,
    convert_attribute: Optional[bool] = False,
) -> bool:
    """Save the ONNX model to the specified path.

    :param model: The ONNX model to save.
    :param output_path: The path to save the ONNX model to.
    :param save_as_external_data: If True, save tensor data to separate files instead of directly in the ONNX file.
        Large models (>2GB) may be forced to save external data regardless of the value of this parameter.
    :param all_tensors_to_one_file: Effective only if save_as_external_data is True. If True, save all tensors to one
        external file specified by 'external_data_name'. If False, save each tensor to a file named with the tensor
        name.
    :param external_data_name: Effective only if all_tensors_to_one_file is True and save_as_external_data is True.
        If not specified, the external data file will be named with <model_path_name>.data

    :return: True if the model has external data, False otherwise.
    """
    output_path = Path(resolve_onnx_path(output_path))
    if output_path.exists():
        logger.info("Deleting existing onnx file: %s", output_path)
        output_path.unlink()

    # parent directory of .onnx file
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not save_as_external_data:
        try:
            # save model
            onnx.save_model(model, str(output_path))
            return False
        except ValueError as e:
            # there are different types of error message for large model (>2GB) based on onnx version
            # just try to save as external data
            # if it fails, raise the original error
            try:
                logger.debug("Model save failed with error: %s. Trying to save as external data.", e)
                model_proto_to_file(model, output_path, True, all_tensors_to_one_file, external_data_name)
                logger.warning(
                    "Model is too large to save as a single file but 'save_as_external_data' is False. Saved tensors"
                    " as external data regardless."
                )
                return True
            except Exception:
                raise e from None

    # location for external data
    external_data_path = output_dir / (external_data_name if external_data_name else f"{output_path.name}.data")
    location = external_data_path.name if all_tensors_to_one_file else None

    if all_tensors_to_one_file:
        if external_data_path.exists():
            # Delete the external data file. Otherwise, data will be appended to existing file.
            logger.info("Deleting existing external data file: %s", external_data_path)
            external_data_path.unlink()
    else:
        if any(output_dir.iterdir()):
            raise RuntimeError(f"Output directory ({output_dir}) for external data is not empty.")

    # save model
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=location,
        size_threshold=size_threshold,
        convert_attribute=convert_attribute,
    )
    return True
