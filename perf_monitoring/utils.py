import json


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
    compared_new_input_metric = compare_input_metrics(best_metrics, model_name)
    print("Compared new input metrics: ", compared_new_input_metric)


def no_regression(actual, expected, rel_tol, higher_is_better):  # check for tolerance
    if higher_is_better and actual > expected:
        return True
    elif not higher_is_better and actual < expected:
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
        latency_percentage_change = -((best_metrics[1] - model_data[1]) / model_data[1]) * 100

        comparison_result = {
            "accuracy": no_regression(best_metrics[0], model_data[0], 0.09, True),
            "latency": no_regression(best_metrics[1], model_data[1], 0.095, False),
            "accuracy_percentage_change": accuracy_percentage_change,
            "latency_percentage_change": latency_percentage_change,
        }

        # Assert that both accuracy and latency are True
        assert comparison_result["accuracy"], "accuracy must be True"
        assert comparison_result["latency"], "latency must be True"

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


def compare_input_metrics(best_metrics, model_name):
    # open best metrics json
    with open(f"models/{model_name}_workflow_cpu/cpu-cpu_input_model_metrics.json") as f:
        data = json.load(f)
    if "accuracy-accuracy" in data:
        accuracy = data["accuracy-accuracy"]["value"]
    else:
        accuracy = data["accuracy-accuracy_score"]["value"]
    # accuracy = data["accuracy-accuracy"]["value"]
    latency = data["latency-avg"]["value"]
    accuracy_percentage_change = ((best_metrics[0] - accuracy) / accuracy) * 100
    latency_percentage_change = -((best_metrics[1] - latency) / latency) * 100

    comparison_result = {
        "accuracy": no_regression(best_metrics[0], accuracy, 0.09, True),
        "latency": no_regression(best_metrics[1], latency, 0.095, False),
        "accuracy_percentage_change": accuracy_percentage_change,
        "latency_percentage_change": latency_percentage_change,
    }
    return comparison_result
