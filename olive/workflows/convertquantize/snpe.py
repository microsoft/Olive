# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import tempfile
from pathlib import Path

from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.snpe import SNPEConversion, SNPEQuantization
from olive.snpe import SNPEProcessedDataLoader
from olive.workflows.convertquantize.config import ConvertQuantizeConfig
from olive.workflows.convertquantize.utils import get_model, prepare_snpe_quant_data, resolve_model_dir

logger = logging.getLogger(__name__)


def snpe_convertquantize(config: ConvertQuantizeConfig):
    model = get_model(config.model)

    models_dir, name = resolve_model_dir(config.model, config.output_dir, config.output_name)

    # ------------------------------------------------------------------
    # SNPE model
    logger.info("Converting model to SNPE...")
    snpe_model_file = str(models_dir / f"{name}_snpe.dlc")

    convert_options = config.convert_options or {}
    snpe_conversion = create_pass_from_dict(
        SNPEConversion, {**config.io_config.dict(), **convert_options}, disable_search=True
    )
    snpe_model = snpe_conversion.run(model, snpe_model_file)
    assert Path(snpe_model.model_path).is_file()
    json.dump(
        snpe_model.io_config, open(str(models_dir / f"{Path(snpe_model_file).stem}.io_config.json"), "w"), indent=2
    )

    # ------------------------------------------------------------------
    # SNPE Quantized model
    if config.input_list_file is not None:
        logger.info("Input list file provided, skipping SNPE quantization data preparation...")
        quant_data_dir = str(config.quant_data.resolve())
        input_list_file = config.input_list_file
    else:
        logger.info("Preparing SNPE quantization data...")
        if config.workspace is not None:
            workspace = config.workspace / "olive-snpe"
        else:
            tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")
            workspace = Path(tmp_dir.name).resolve()
        quant_data_dir = str(workspace)
        input_list_file = prepare_snpe_quant_data(
            config.quant_data, config.io_config.dict(), snpe_model.io_config, workspace
        )

    logger.info("Quantizing SNPE model...")
    snpe_quantized_model_file = str(models_dir / f"{name}_snpe_quantized.dlc")
    dataloader_func = lambda data_dir: SNPEProcessedDataLoader(data_dir, input_list_file=input_list_file)  # noqa: E731

    quantize_options = config.quantize_options or {}
    snpe_quantization = create_pass_from_dict(
        SNPEQuantization,
        {"data_dir": quant_data_dir, "dataloader_func": dataloader_func, **quantize_options},
        disable_search=True,
    )
    snpe_quantized_model = snpe_quantization.run(snpe_model, snpe_quantized_model_file)
    assert Path(snpe_quantized_model.model_path).is_file()
    json.dump(
        snpe_quantized_model.io_config,
        open(str(models_dir / f"{Path(snpe_quantized_model_file).stem}.io_config.json"), "w"),
        indent=2,
    )
