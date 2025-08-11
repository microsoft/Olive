# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path

from olive.engine.output import WorkflowOutput

# pylint: disable=broad-exception-raised, W0212


def check_output(workflow_output: WorkflowOutput):
    """Check if the search output is valid."""
    assert_nodes(workflow_output)
    assert_metrics(workflow_output)


def assert_nodes(workflow_output: WorkflowOutput):
    assert workflow_output, "workflow_output is empty. The search must have failed for all accelerator specs."
    assert workflow_output.has_output_model(), "No output model found."


def assert_metrics(workflow_output: WorkflowOutput):
    for output_model in workflow_output.get_output_models():
        assert all(metric_result.value > 0 for metric_result in output_model._model_node.metrics.value.values()), (
            "No metrics found."
        )


def patch_config(config_json_path: str, sampler: str = None, execution_order: str = None):
    """Load the config json file and patch it with the given search algorithm, execution order and system."""
    with open(config_json_path) as fin:
        olive_config = json.load(fin)
    # set default logger severity
    olive_config["log_severity_level"] = 0
    # set clean cache
    olive_config["clean_cache"] = True

    # update search strategy
    if not sampler:
        olive_config["search_strategy"] = False
    else:
        olive_config["search_strategy"] = {
            "sampler": sampler,
            "execution_order": execution_order,
        }
        if sampler in ("random", "tpe"):
            olive_config["search_strategy"].update({"max_samples": 3, "seed": 0})

    return olive_config


def get_example_dir(example_name: str):
    return str(Path(__file__).resolve().parent.parent / example_name)
