import json


def check_search_output(footprints):
    """Check if the search output is valid."""
    assert footprints, "footprints is empty. The search must have failed for all accelerator specs."
    for footprint in footprints.values():
        assert footprint.nodes
        for v in footprint.nodes.values():
            assert all([metric_result.value > 0 for metric_result in v.metrics.value.values()])


def check_no_search_output(outputs):
    assert outputs, "outputs is empty. The run must have failed for all accelerator specs."
    for output in outputs.values():
        output_metrics = output["metrics"]
        for item in output_metrics.values():
            assert item.value > 0


def patch_config(config_json_path: str):
    """Load the config json file and patch it with default search algorithm (exhaustive)"""
    with open(config_json_path, "r") as fin:
        olive_config = json.load(fin)
    # set default logger severity
    olive_config["engine"]["log_severity_level"] = 0
    # set clean cache
    olive_config["engine"]["clean_cache"] = True
    return olive_config
