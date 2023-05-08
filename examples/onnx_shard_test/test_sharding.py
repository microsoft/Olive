from olive.systems.local import LocalSystem
from olive.model import ONNXModel, DistributedOnnxModel
from olive.evaluator.metric import LatencySubType, Metric, MetricType
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.engine import Engine
from olive.passes import OnnxModelSharding
from olive.passes.olive_pass import create_pass_from_dict
import sys
from olive.evaluator.distributed_evaluator import eval_onnx_distributed_latency, _mpipool_worker


def _main():
    input_model = ONNXModel(
        model_path="bloom-test-model_opt.onnx"
    )

    # create latency metric instance
    latency_metric = Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_type=LatencySubType.AVG,
        user_config={
            "user_script": "inference_script.py",
            "data_dir": "data",
            "evaluate_func": "eval_latency",
            "batch_size": 1,
        }
    )

    # create evaluator
    evaluator = OliveEvaluator(metrics=[latency_metric])

    # configuration options for engine
    engine_config = {
        "cache_dir": ".cache"
    }

    local_system = LocalSystem()

    engine = Engine(engine_config, evaluator=evaluator, host=local_system)

    onnx_conversion_config = {
        "sharding_spec_path": "best_shard.json",
        "hardware_spec_path": "hardware-test-device.json"
    }

    onnx_sharding_pass = create_pass_from_dict(OnnxModelSharding, onnx_conversion_config)

    # override the default host with pass specific host
    engine.register(onnx_sharding_pass, host=LocalSystem())

    best_execution = engine.run(input_model, verbose=True)


if __name__ == "__main__":
    sys.exit(_main())
