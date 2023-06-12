# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

from tabulate import tabulate


def get_directories():
    current_dir = Path(__file__).resolve().parent

    # models directory for resnet sample
    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # data directory for resnet sample
    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, models_dir, data_dir


def print_metrics(metrics_dict, metric_headers):
    """Print metrics in a table format.

    Args:
        metrics_dict (dict): Dictionary of metrics for each model.
            {"model_name": metrics}
        metric_headers (dict): Dictionary of metric headers.
            {"metric_name": "metrics_key")}
    """
    headers = ["Model"] + list(metric_headers.keys())
    values = []
    for model_name, metrics in metrics_dict.items():
        values.append([model_name] + [metrics.get(name, "NA") for name in metric_headers.values()])
    print("Metrics:")
    print(tabulate(values, headers=headers, tablefmt="github", floatfmt=".4f"))
