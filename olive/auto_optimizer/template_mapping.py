# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

TEMPLATE_CONFIG_PATH = Path(__file__).resolve().parent / "config_template"
AVAILABLE_PASS_FLOWS = "available_pass_flows.yaml"
PRECISION_AWARE_PASS_FLOWS = "precision_aware_pass_flows.yaml"


def get_precision_aware_passes():
    with (TEMPLATE_CONFIG_PATH / PRECISION_AWARE_PASS_FLOWS).open() as f:
        return yaml.safe_load(f)


def get_available_pass_flows():
    """Get all the available pass flows from the yaml file.

    return: a dict of pass flows, key is the combination of accelerator, ep and precision.
        value is a list of pass flows.
        e.g. for accelerator: "GPU", ep: "CUDAExecutionProvider", precision: "fp32", the key is "gpu_cuda_fp32"
    """
    with (TEMPLATE_CONFIG_PATH / AVAILABLE_PASS_FLOWS).open() as f:
        available_pass_flows = yaml.safe_load(f)
    return available_pass_flows["mapping"]


def get_pass_flows_by_accelerator_ep_precision(accelerator, ep, precision):
    ep_literal = "ExecutionProvider"
    ep = ep[: -len(ep_literal)] if ep.endswith(ep_literal) else ep
    pass_flows_key = f"{accelerator.lower()}_{ep.lower()}_{precision.lower()}"
    available_pfs = get_available_pass_flows()
    if pass_flows_key not in available_pfs:
        logger.debug(f"pass flows for {pass_flows_key} is not in {available_pfs}, will ignore it")
        return []
    return available_pfs[pass_flows_key]
