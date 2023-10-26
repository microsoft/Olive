# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from itertools import product
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

TEMPLATE_CONFIG_PATH = Path(__file__).resolve().parent / "config_template"
DEFAULT_ONNX_OPT_STACK = "onnx_pass_flows.yaml"


def default_onnx_opt_pass_flows():
    with (TEMPLATE_CONFIG_PATH / DEFAULT_ONNX_OPT_STACK).open() as f:
        default_pass_groups = yaml.safe_load(f)
    # extend the pass groups to pass flows
    default_pass_flows = []
    pass_groups = default_pass_groups["pass_groups"]
    pass_flow_groups = default_pass_groups["pass_flow_groups"]

    for pf_group in pass_flow_groups:
        candidate_list = []
        for candidate in pf_group:
            candidate_list.append(pass_groups[candidate]["passes"])
        for pf in product(*candidate_list):
            default_pass_flows.append(list(pf))

    return default_pass_flows
