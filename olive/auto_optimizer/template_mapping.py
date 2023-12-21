# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from collections import deque
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

TEMPLATE_CONFIG_PATH = Path(__file__).resolve().parent / "config_template"
PASS_CAPABILITY = "pass_capability.yaml"
OPT_LEVEL_PASSES = "opt_level_passes.yaml"


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
    precision = precision.lower()

    available_passes_tree = get_available_passes_by_opt_level(opt_level)
    passes_cap = get_pass_capability()
    pass_flows = []

    # given available_passes_tree is [a] -> [b, c] -> [d, e, f], generate all possible pass flows
    # [a, b, d], [a, b, e], [a, b, f], [a, c, d], [a, c, e], [a, c, f]
    # as we need to step over some intermediate passes, we cannot use len(pass_flow_candidate) to
    # indicate the current pass level, instead, we use the length of available_passes_tree to indicate
    # item in pass stack is (pass, depth, pass_flow_candidate)
    pass_deque = deque([(-1, [])])
    while pass_deque:
        depth, pf_candidate = pass_deque.popleft()

        # strong rule when met the last pass, traverse back to the previous pass
        if depth == len(available_passes_tree) - 1:
            pass_flows.append(pf_candidate)
            continue

        # if we don't have any pass in next level, we cannot step over it
        keep_try = True
        for next_level in range(depth + 1, len(available_passes_tree)):
            if keep_try:
                for p_next in available_passes_tree[next_level]:
                    if _if_match_pass_capability(p_next, passes_cap, accelerator, ep, precision):
                        pass_deque.append((next_level, [*pf_candidate, p_next]))
                        # if we find one pass in next_level, break the outer loop
                        keep_try = False
            # not `elif` here, as we need to check special case for fp16
            if not keep_try:
                if precision == "fp16" and len(available_passes_tree[next_level]) > 1:
                    # for fp16, we can step over to next level + 1 even we find one pass in next level
                    keep_try = True
                else:
                    break

    return pass_flows


def _if_match_pass_capability(p, passes_cap, accelerator, ep, precision):
    # get support precisions for this pass
    if (lower_ep := passes_cap[p].get("EP")) is not None:
        lower_ep = [e.lower() for e in lower_ep]
    if (
        # if accelerator & precision & ep are matched simultaneously, added it into candidates
        (passes_cap[p].get("accelerator") is None or accelerator in passes_cap[p].get("accelerator"))
        and (lower_ep is None or ep in lower_ep)
        and (passes_cap[p].get("precision") is None or precision in passes_cap[p].get("precision"))
    ):
        return True
    return False
