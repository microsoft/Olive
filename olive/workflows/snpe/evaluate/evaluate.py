# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from olive.hardware import Device
from olive.model import SNPEModel
from olive.snpe import SNPEProcessedDataLoader
from olive.snpe.utils.local import get_snpe_target_arch

logger = logging.getLogger(__name__)


def evaluate(model: str, config: Union[str, Dict], data: str, input_list_file: Optional[str] = "input_list.txt"):
    """Evaluate a model.

    Args:
        model (str): Path to the model.
        config (str): Either the path of json config file or an already loaded json file as a `dict`.
        data (str): Path to the evaluation data.
        input_list_file (str, optional): Name of input list file. Optional if it is 'input_list.txt'.
    """
    data_dir = Path(data).resolve()
    if isinstance(config, str):
        with Path(config).resolve().open() as f:
            config = json.load(f)

    # SNPE Model
    model = SNPEModel(model_path=model, **config["io_config"])

    # Devices to evaluate on
    devices = [Device.CPU]
    if get_snpe_target_arch() == "ARM64-Windows":
        devices.append(Device.NPU)

    config["inference_settings"]["return_numpy_results"] = True
    # config["inference_settings"]["android_target"] = "f85154f6"
    # devices.append(Device.NPU)

    # Data
    data = SNPEProcessedDataLoader(data_dir, input_list_file=input_list_file)
    input_list = data.get_input_list()

    # Run inference
    results = []
    for device in devices:
        logger.info(f"Running inference on {device}...")
        session = model.prepare_session(config["inference_settings"], device=device)
        out = session(input_list, data_dir)
        logger.info(
            "Latencies:\n"
            f"Init: {round(out['latencies']['init'][0] * 1000, 3)} ms\n"
            f"Total Inference: {round(out['latencies']['total_inference_time'][0] * 1000, 3)} ms"
        )
        results.append(out["results"])
        throughput = session.throughput(5, input_list, data_dir)
        logger.info(f"Throughput: {throughput} infs/sec")

    # Compare results
    if len(results) > 1:
        avg_distances = {}
        for key in results[0]:
            batch_size = results[0][key].shape[0]
            results_0 = results[0][key].reshape(batch_size, -1)
            results_1 = results[1][key].reshape(batch_size, -1)
            avg_distances[key] = np.linalg.norm(results_0 - results_1, axis=1)
            avg_distances[key] = avg_distances[key].mean()
        logger.info(f"Average distances between outputs: {avg_distances}")
