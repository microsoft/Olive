# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from typing import Dict, Union

from olive.common.config_utils import validate_config
from olive.logging import set_verbosity
from olive.workflows.convertquantize.config import ConvertQuantizeConfig, logging_verbosity
from olive.workflows.convertquantize.snpe import snpe_convertquantize


def convertquantize(config: Union[str, Dict, ConvertQuantizeConfig]):
    if type(config) is str:
        config = json.load(open(Path(config).resolve()))
    config = validate_config(config, ConvertQuantizeConfig)

    # set logging verbosity
    set_verbosity(logging_verbosity[config.verbosity])

    if config.tool == "snpe":
        snpe_convertquantize(config)
    elif config.tool == "openvino":
        from olive.workflows.convertquantize.openvino import openvino_convertquantize

        openvino_convertquantize(config)
    else:
        raise ValueError(f"Unsupported tool: {config.tool}")
