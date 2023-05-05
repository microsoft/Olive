# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from typing import Dict, Union

from olive.common.config_utils import validate_config
from olive.logging import set_verbosity
from olive.workflows.wssi.config import ConvertQuantizeConfig, logging_verbosity
from olive.workflows.wssi.snpe import snpe_convertquantize


def convertquantize(config: Union[str, Dict, ConvertQuantizeConfig]):
    if type(config) is str:
        config = json.load(open(config))
    config = validate_config(config, ConvertQuantizeConfig)

    # set logging verbosity
    set_verbosity(logging_verbosity[config.verbosity])

    if config.tool == "snpe":
        snpe_convertquantize(config)
    elif config.tool == "openvino":
        from olive.workflows.wssi.openvino import openvino_convertquantize

        openvino_convertquantize(config)
    else:
        raise ValueError(f"Unsupported tool: {config.tool}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Olive Workflow: ConvertQuantize")
    parser.add_argument("--config", type=str, help="Path to json config file", required=True)

    args = parser.parse_args()

    convertquantize(**vars(args))
