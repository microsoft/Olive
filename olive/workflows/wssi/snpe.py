# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from olive.model import SNPEModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.snpe import SNPEConversion, SNPEQuantization
from olive.snpe import SNPEProcessedDataLoader
from olive.snpe.utils.local import get_snpe_target_arch
from olive.systems.common import Device
from olive.workflows.wssi.config import ConvertQuantizeConfig
from olive.workflows.wssi.utils import get_model, prepare_snpe_quant_data, resolve_model_dir

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
    if config.workspace is not None:
        workspace = config.workspace / "olive-snpe"
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")
        workspace = Path(tmp_dir.name).resolve()
    quant_data_dir, input_list_file = get_processed_data(
        workspace, config.quant_data, config.input_list_file, config.io_config.dict(), snpe_model.io_config
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


def snpe_evaluate(
    config: ConvertQuantizeConfig,
    model: Union[str, Path],
    io_config: Union[str, Path],
    eval_data: Optional[Union[str, Path]] = None,
    input_list_file: Optional[Union[str, Path]] = None,
):
    snpe_io_config = json.load(open(io_config, "r"))
    snpe_model = SNPEModel(model_path=model, **snpe_io_config)

    # evaluation data
    # use quant data if eval data is not provided
    if eval_data:
        input_list_file = input_list_file
    else:
        eval_data = config.quant_data
        input_list_file = input_list_file or config.input_list_file

    # workspace
    tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")
    workspace = Path(tmp_dir.name).resolve()

    # processed data
    eval_data_dir, input_list_file = get_processed_data(
        workspace, eval_data, input_list_file, config.io_config.dict(), snpe_io_config
    )

    # Devices to evaluate on
    devices = [Device.CPU]
    if get_snpe_target_arch() == "ARM64-Windows":
        devices.append(Device.NPU)

    # snpe inference settings
    inference_settings = {
        "set_output_tensors": len(snpe_io_config["output_names"]) > 1,
        "perf_profile": "sustained_high_performance",
        "profiling_level": "moderate",
        "return_numpy_results": True,
    }

    # dataloader
    data = SNPEProcessedDataLoader(eval_data_dir, input_list_file=input_list_file)
    input_list = data.get_input_list()

    # Run inference
    results = []
    for device in devices:
        logger.info(f"Running inference on {device}...")
        session = snpe_model.prepare_session(inference_settings, device=device)
        out = session(input_list, eval_data_dir)
        logger.info(
            "Latencies:\n"
            f"Init: {round(out['latencies']['init'][0] * 1000, 3)} ms\n"
            f"Total Inference: {round(out['latencies']['total_inference_time'][0] * 1000, 3)} ms"
        )
        results.append(out["results"])
        throughput = session.throughput(5, input_list, eval_data_dir)
        logger.info(f"Throughput: {throughput} infs/sec")

    # Compare results between devices
    if len(results) > 1:
        avg_distances = {}
        for key in results[0]:
            batch_size = results[0][key].shape[0]
            results_0 = results[0][key].reshape(batch_size, -1)
            results_1 = results[1][key].reshape(batch_size, -1)
            avg_distances[key] = np.linalg.norm(results_0 - results_1, axis=1)
            avg_distances[key] = avg_distances[key].mean()
        logger.info(f"Average distances between outputs: {avg_distances}")


def get_processed_data(
    workspace: Union[str, Path],
    data: Union[str, Path],
    input_list_file: Union[str, Path],
    io_config: dict,
    snpe_io_config: dict,
):
    if input_list_file:
        logger.info("Input list file provided, skipping SNPE data preparation...")
        return str(Path(data).resolve()), str(Path(input_list_file).resolve())

    logger.info("Preparing SNPE data...")
    quant_data_dir = str(workspace)
    input_list_file = prepare_snpe_quant_data(data, io_config, snpe_io_config, workspace)
    return quant_data_dir, input_list_file
