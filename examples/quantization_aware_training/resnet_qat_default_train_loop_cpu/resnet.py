# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import warnings

import onnxruntime as ort
from resnet_utils import get_directories

from olive.engine import Engine
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.model import PyTorchModel
from olive.passes import OnnxConversion, OrtPerfTuning, QuantizationAwareTraining
from olive.systems.local import LocalSystem

warnings.simplefilter(action="ignore", category=FutureWarning)

ort.set_default_logger_severity(3)


def get_args():
    parser = argparse.ArgumentParser(description="Olive vnext bert qat example")
    parser.add_argument("--gpu", action="store_true", help="Run evaluation on GPU")
    parser.add_argument(
        "--optimize_metric",
        type=str,
        choices=["accuracy", "latency"],
        default="accuracy",
        help="Metric to optimize for: accuracy or latency",
    )
    parser.add_argument(
        "--search_algorithm",
        type=str,
        choices=["exhaustive", "random", "tpe"],
        default="exhaustive",
        help="Search algorithm: exhaustive or random",
    )
    parser.add_argument(
        "--execution_order",
        type=str,
        choices=["joint", "pass-by-pass"],
        default="pass-by-pass",
        help="Execution order: joint or pass-by-pass",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # directories
    current_dir, models_dir, _, cache_dir = get_directories()
    user_script = str(current_dir / "user_script.py")
    name = "resnet_trained_for_cifar10"

    # ------------------------------------------------------------------
    # Evaluator
    accuracy_metric_config = {
        "user_script": user_script,
        "post_processing_func": "post_process",
        "dataloader_func": "create_benchmark_dataloader",
        "batch_size": 1,
    }
    accuracy_metric = Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_type=AccuracySubType.ACCURACY_SCORE,
        user_config=accuracy_metric_config,
    )

    latency_metric_config = {
        "user_script": user_script,
        "dataloader_func": "create_benchmark_dataloader",
        "batch_size": 1,
    }
    latency_metric = Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_type=LatencySubType.AVG,
        higher_is_better=False,
        user_config=latency_metric_config,
    )
    metrics = [accuracy_metric, latency_metric]
    evaluator = OliveEvaluator(metrics=metrics, target=LocalSystem())

    # ------------------------------------------------------------------
    # Engine
    options = {
        "cache_dir": str(cache_dir),
        "search_strategy": {
            "execution_order": args.execution_order,
            "search_algorithm": args.search_algorithm,
        },
    }
    engine = Engine(options, evaluator=evaluator)

    # ------------------------------------------------------------------
    # Quantization Aware Training pass
    qat_config = {
        "user_script": user_script,
        "input_shapes": [[1, 3, 32, 32]],
        "num_epochs": 2,
        "train_dataloader_func": "create_train_dataloader",
        "train_batch_size": 100,
        "modules_to_fuse": [["conv1", "bn1"], ["conv2", "bn2"], ["conv3", "bn3"]],
        "qconfig_func": "create_qat_config",
        "seed": 42,
    }
    qat_pass = QuantizationAwareTraining(qat_config, disable_search=True)
    engine.register(qat_pass)

    # ------------------------------------------------------------------
    # Onnx conversion pass
    # config can be a dictionary
    onnx_conversion_config = {
        "input_names": ["input"],
        "input_shapes": [[1, 3, 32, 32]],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        "target_opset": 17,
    }
    onnx_conversion_pass = OnnxConversion(onnx_conversion_config, disable_search=True)
    engine.register(onnx_conversion_pass)

    # ------------------------------------------------------------------
    # ONNX Runtime performance tuning pass
    ort_perf_tuning_config = {
        "user_script": user_script,
        "dataloader_func": "create_benchmark_dataloader",
        "batch_size": 1,
    }
    ort_perf_tuning_pass = OrtPerfTuning(ort_perf_tuning_config)
    engine.register(ort_perf_tuning_pass)

    # ------------------------------------------------------------------
    # Input model
    pytorch_model_file = str(models_dir / f"{name}.pt")
    pytorch_model = PyTorchModel(pytorch_model_file, is_file=True)

    # ------------------------------------------------------------------
    # Run engine
    best_execution = engine.run(pytorch_model, verbose=True)
    print(best_execution)

    return best_execution["metric"]


if __name__ == "__main__":
    main()
