# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse

from inception_utils import get_directories, print_metrics

from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.model import SNPEModel
from olive.systems.common import Device
from olive.systems.local import LocalSystem


def get_args():
    parser = argparse.ArgumentParser(description="Olive vnext Inception example")

    parser.add_argument("--use_dsp", action="store_true", help="Use DSP when evaluating quantized model")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    current_dir, models_dir, data_dir = get_directories()
    user_script = str(current_dir / "user_script.py")
    data_folder = str(data_dir)
    name = "inception_v3"
    io_config = {
        "input_names": ["input"],
        "input_shapes": [[1, 299, 299, 3]],
        "output_names": ["InceptionV3/Predictions/Reshape_1"],
        "output_shapes": [[1, 1001]],
    }
    inference_settings = {
        "snpe": {
            "return_numpy_results": True,
            "set_output_tensors": False,
            "perf_profile": "sustained_high_performance",
            "profiling_level": "moderate",
            # "android_target": "f85154f6",
            # "android_target": "emulator-5554",
            # "workspace": str(current_dir / "workspace"),
        }
    }

    # ------------------------------------------------------------------
    # SNPE model
    snpe_model_file = str(models_dir / f"{name}_snpe.dlc")
    snpe_model = SNPEModel(model_path=snpe_model_file, name=name, **io_config)

    # ------------------------------------------------------------------
    # SNPE Quantized model
    snpe_quantized_model_file = str(models_dir / f"{name}_snpe_quantized.dlc")
    snpe_quantized_model = SNPEModel(model_path=snpe_quantized_model_file, name=name, **io_config)

    # ------------------------------------------------------------------
    # Models dictionary
    models_dict = {"SNPE": snpe_model, "SNPE Quantized": snpe_quantized_model}

    # ------------------------------------------------------------------
    # Evaluate models
    print("Evaluating models:")
    # Define metrics
    accuracy_metrics = Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_type=AccuracySubType.ACCURACY_SCORE,
        user_config={
            "user_script": user_script,
            "post_processing_func": "post_process",
            "data_dir": data_folder,
            "batch_size": 7,
            "dataloader_func": "create_eval_dataloader",
            "inference_settings": inference_settings,
        },
    )
    latency_metrics = Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_type=LatencySubType.AVG,
        user_config={
            "user_script": user_script,
            "data_dir": data_folder,
            "batch_size": 7,
            "dataloader_func": "create_eval_dataloader",
            "inference_settings": inference_settings,
        },
        metric_config={"warmup_num": 0, "repeat_test_num": 5, "sleep_num": 2},
    )

    # Evaluate models
    devices_dict = {
        "SNPE": Device.CPU,
        "SNPE Quantized": Device.NPU if args.use_dsp else Device.CPU,
    }
    metrics_dict = {}
    for model_name in models_dict:
        device = devices_dict[model_name]
        system = LocalSystem(device=device)
        print(f"   {model_name} on {device}...")
        metrics = system.evaluate_model(models_dict[model_name], [accuracy_metrics, latency_metrics])
        metrics_dict[model_name] = metrics

    # Print metrics
    metric_headers = {"Accuracy": "accuracy", "Average Latency (ms)": "latency"}

    print_metrics(metrics_dict, metric_headers)

    return metrics_dict


if __name__ == "__main__":
    main()
