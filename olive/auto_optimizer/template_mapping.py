# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

TEMPLATE_CONFIG_PATH = Path(__file__).resolve().parent / "config_template"
# precision_aware_pass_flows: means the passes which can conduct different precision models
# like: transformer optimization and perf tuning(trt fp16)
PRECISION_AWARE_PASS_FLOWS = "precision_aware_pass_flows.yaml"
PASS_CAPABILITY = "pass_capability.yaml"
OPT_LEVEL_PASSES = "opt_level_passes.yaml"

LEADING_PASSES = ["OnnxConversion", "OrtTransformersOptimization"]
TRAILING_PASSES = ["OrtPerfTuning"]


def get_precision_aware_passes():
    with (TEMPLATE_CONFIG_PATH / PRECISION_AWARE_PASS_FLOWS).open() as f:
        return yaml.safe_load(f)


def get_pass_capability():
    with (TEMPLATE_CONFIG_PATH / PASS_CAPABILITY).open() as f:
        return yaml.safe_load(f)


def get_available_passes_by_opt_level(opt_level):
    with (TEMPLATE_CONFIG_PATH / OPT_LEVEL_PASSES).open() as f:
        opt_level_passes = yaml.safe_load(f)
    return opt_level_passes[str(opt_level)]


def get_pass_flows_by_accelerator_ep_precision(opt_level, accelerator, ep, precision):
    ep_literal = "ExecutionProvider"
    ep = ep[: -len(ep_literal)].lower() if ep.endswith(ep_literal) else ep.lower()
    available_passes = get_available_passes_by_opt_level(opt_level)
    passes_cap = get_pass_capability()
    passes_candidates = []
    passes_candidates_config = {}
    for p in available_passes:
        if p in LEADING_PASSES or p in TRAILING_PASSES:
            continue
        # None means all, means no constraints
        if (lower_ep := passes_cap[p].get("EP")) is not None:
            lower_ep = [e.lower() for e in lower_ep]
        if (
            (passes_cap[p].get("accelerator") is None or accelerator in passes_cap[p].get("accelerator"))
            and (lower_ep is None or ep in lower_ep)
            and (passes_cap[p].get("precision") is None or precision in passes_cap[p].get("precision"))
        ):
            passes_candidates.append(p)
            passes_candidates_config[p] = {
                "accelerator": None if passes_cap[p].get("accelerator") is None else accelerator,
                "EP": None if passes_cap[p].get("EP") is None else ep,
                "precision": None if passes_cap[p].get("precision") is None else precision,
            }

    # assumption: only one pass whose precision is not fp32 or None can appear in a pass flow
    # for example:
    # 1. OnnxQuantization and OnnxMatMul4Quantizer cannot appear in the same pass
    # flow as we cannot quantize a quantized model
    # 2. OnnxQuantization and OrtTransformersOptimization_fp16 can not appear in
    # the same pass flow as we should avoid to quantize a fp16 model
    # etc.

    pass_flows = []
    for p in passes_candidates:
        pass_flows.append([*LEADING_PASSES, p, *TRAILING_PASSES])
    if not pass_flows or precision == "fp16":
        pass_flows.append([*LEADING_PASSES, *TRAILING_PASSES])
    return pass_flows
