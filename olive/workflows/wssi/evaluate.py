# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path
from typing import Optional, Union

from olive.common.config_utils import validate_config
from olive.logging import set_verbosity
from olive.workflows.wssi.config import ConvertQuantizeConfig, logging_verbosity
from olive.workflows.wssi.snpe import snpe_evaluate


def evaluate(
    config: ConvertQuantizeConfig,
    model: Union[str, Path],
    io_config: Optional[Union[str, Path]] = None,
    eval_data: Optional[Union[str, Path]] = None,
    input_list_file: Optional[Union[str, Path]] = None,
):
    if type(config) is str:
        config = json.load(open(config))
    config = validate_config(config, ConvertQuantizeConfig)

    # set logging verbosity
    set_verbosity(logging_verbosity[config.verbosity])

    if config.tool == "snpe":
        snpe_evaluate(config, model, io_config, eval_data, input_list_file)
    elif config.tool == "openvino":
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported tool: {config.tool}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: ConvertQuantize")
    parser.add_argument("--config", type=str, help="Path to json config file.", required=True)
    parser.add_argument("--model", type=str, help="Path to model file.", required=True)
    parser.add_argument(
        "--io_config", type=str, default=None, help="Path to model's IO config file. Only required for SNPE models."
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help=(
            "Path to evaluation data. If not provided, use the quantization data. For SNPE models, if input list is not"
            " provided, it is assumed that the data is in the same shape as the original model. Data will be processed"
            " if needed."
        ),
    )
    parser.add_argument(
        "--input_list_file",
        type=str,
        default=None,
        help=(
            "Path to input list file. Only applicable for SNPE models. If not provided, the data will be processed and"
            " the input list file will be created during the evaluation."
        ),
    )

    args = parser.parse_args()

    evaluate(**vars(args))
