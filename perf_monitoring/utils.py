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


def extract_best_models(footprint, model_name):
    print("Footprint: ", footprint)
    footprint = list(footprint.values())[0]
    print(
        "Footprint: ",
        footprint,
    )
    metrics_of_interest = ["accuracy-accuracy", "accuracy-accuracy_score", "latency-avg"]
    # gather the metrics from all pareto frontier nodes
    all_metrics = []
    # we iterate over the nodes in the pareto frontier
    for node in footprint.nodes.values():
        metrics = []
        # collecting the metrics of interest
        for name in metrics_of_interest:
            # (value of metric * direction of comparison)
            # now higher is better for all metrics
            if name in node.metrics.value:
                metrics.append(node.metrics.value[name].value * node.metrics.cmp_direction[name])
        all_metrics.append(metrics)
    # sort the metrics
    # this sorts it
    sorted_metrics = sorted(all_metrics, reverse=True)
    # get best metrics
    # last one is the best
    best_metrics = sorted_metrics[0]
    print("Best metrics: ", best_metrics)
    compared_metric = compare_metrics(best_metrics, model_name)
    print("Compared metrics: ", compared_metric)


def no_regression(actual, expected, rel_tol):  # check for tolerance
    if actual > expected:
        return True
    return abs(actual - expected) <= rel_tol * abs(expected)


def compare_metrics(best_metrics, model_name):
    # open best metrics json
    with open("best_metrics.json") as f:
        data = json.load(f)

    if model_name in data:
        model_data = data[model_name]
        if len(model_data) == 0:
            print("No data in best_metrics.json")
            return {"accuracy": True, "latency": True, "accuracy_percentage_change": 0, "latency_percentage_change": 0}
        print(model_data[0], model_data[1])
        print(best_metrics[0], best_metrics[1])

        accuracy_percentage_change = ((best_metrics[0] - model_data[0]) / model_data[0]) * 100
        latency_percentage_change = ((best_metrics[1] - model_data[1]) / model_data[1]) * 100

        comparison_result = {
            "accuracy": no_regression(best_metrics[0], model_data[0], 0.05),
            "latency": no_regression(best_metrics[1], model_data[1], 0.05),
            "accuracy_percentage_change": accuracy_percentage_change,
            "latency_percentage_change": latency_percentage_change,
            "accuracy_better": "same"
            if accuracy_percentage_change == 0
            else "higher"
            if accuracy_percentage_change > 0
            else "lower",
            "latency_better": "same"
            if latency_percentage_change == 0
            else "lower"
            if latency_percentage_change > 0
            else "higher",
        }

        with open("model_output.txt", "w") as f:
            f.write(f"Accuracy percentage change: {accuracy_percentage_change}\n")
            f.write(f"Latency percentage change: {latency_percentage_change}\n")
            f.write(f"Is accuracy better?: {comparison_result['accuracy_better']}\n")
            f.write(f"Is latency better?: {comparison_result['latency_better']}\n")
    else:
        print(f"{model_name} not found in best_metrics.json, creating new entry...")
        data[model_name] = best_metrics
        comparison_result = {
            "accuracy": True,
            "latency": True,
            "accuracy_percentage_change": 0,
            "latency_percentage_change": 0,
        }

    # Save the updated data back to the file
    with open("best_metrics.json", "w") as f:
        json.dump(data, f, indent=4)

    return comparison_result
