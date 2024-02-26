# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from olive.model import ONNXModelHandler, TensorFlowModelHandler
from olive.passes import SNPEConversion, SNPEQuantization, SNPEtoONNXConversion
from olive.passes.olive_pass import create_pass_from_dict
from olive.platform_sdk.qualcomm.constants import SNPEDevice
from olive.platform_sdk.qualcomm.utils.data_loader import FileListProcessedDataLoader

logger = logging.getLogger(__name__)


def convertquantize(
    model: str,
    config: Union[str, Dict],
    data: str,
    input_list_file: Optional[str] = "input_list.txt",
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
):
    """Convert and quantize a model.

    Args:
        model (str): Path to the model.
        config (str): Either the path of json config file or an already loaded json file as a `dict`.
        data (str): Path to the quantization tuning data.
        input_list_file (str, optional): Name of input list file. Optional if it is 'input_list.txt'.
        output_dir (str, optional): Path to the output directory. Optional if it is the same as model directory.
        output_name (str, optional): Name of the output model (without extension). Optional if same as model name.

    """
    models_dir = Path(model).resolve().parent if output_dir is None else Path(output_dir).resolve()
    data_dir = Path(data).resolve()
    name = Path(model).resolve().stem if output_name is None else output_name
    if isinstance(config, str):
        with Path(config).resolve().open() as f:
            config = json.load(f)

    # ------------------------------------------------------------------
    model_file = Path(model).resolve()
    if model_file.suffix == ".onnx":
        logger.info("Loading model...")
        model = ONNXModelHandler(model_file)
    elif model_file.suffix == ".pb":
        logger.info("Loading model...")
        model = TensorFlowModelHandler(model_file)
    else:
        raise ValueError(f"Unsupported model format: {model_file.suffix}")

    # ------------------------------------------------------------------
    # SNPE model
    logger.info("Converting model to SNPE...")
    snpe_model_file = str(models_dir / f"{name}.dlc")

    snpe_conversion = create_pass_from_dict(
        SNPEConversion, {**config["io_config"], **config["convert_options"]}, disable_search=True
    )
    snpe_model = snpe_conversion.run(model, None, snpe_model_file)
    assert Path(snpe_model.model_path).is_file()
    with (models_dir / f"{name}.dlc_io_config.json").open("w") as f:
        json.dump(snpe_model.io_config, f)

    # ------------------------------------------------------------------
    # SNPE Quantized model
    logger.info("Quantizing SNPE model...")
    snpe_quantized_model_file = str(models_dir / f"{name}.quant.dlc")

    def dataloader_func(data_dir):
        return FileListProcessedDataLoader(data_dir, input_list_file=input_list_file)

    snpe_quantization = create_pass_from_dict(
        SNPEQuantization,
        {"data_dir": str(data_dir), "dataloader_func": dataloader_func, **config["quantize_options"]},
        disable_search=True,
    )
    snpe_quantized_model = snpe_quantization.run(snpe_model, None, snpe_quantized_model_file)
    assert Path(snpe_quantized_model.model_path).is_file()
    with (models_dir / f"{name}.quant.dlc_io_config.json").open("w") as f:
        json.dump(snpe_quantized_model.io_config, f)

    # ------------------------------------------------------------------
    # SNPE Quantized ONNX model
    save_to_onnx = config.get("save_to_onnx", False)
    if not save_to_onnx:
        return

    logger.info("Converting SNPE Quantized model to ONNX...")
    snpe_quantized_onnx_model_file = str(models_dir / f"{name}.quant.onnx")

    snpe_to_onnx_conversion = create_pass_from_dict(
        SNPEtoONNXConversion,
        {"target_device": config["quantize_options"].get("target_device", SNPEDevice.CPU)},
        disable_search=True,
    )
    snpe_quantized_onnx_model = snpe_to_onnx_conversion.run(snpe_quantized_model, None, snpe_quantized_onnx_model_file)
    assert Path(snpe_quantized_onnx_model.model_path).is_file()
