import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def get_path():
    return str(Path(__file__).resolve())


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument("--model_path", type=str, required=True, help="Path to onnx model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output")
    parser.add_argument("--device", type=str, required=True, help="Device to run on")
    parser.add_argument("--type", type=str, required=True, choices=["accuracy", "latency"], help="Type of metric")
    parser.add_argument(
        "--warmup_num", type=int, default=10, help="Number of warmup iterations. Only for latency metric"
    )
    parser.add_argument("--repeat_test_num", type=int, default=20, help="Number of iterations. Only for latency metric")
    parser.add_argument("--sleep_num", type=int, default=0, help="Number of sleep iterations. Only for latency metric")

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # create session
    if args.device == "cpu":
        backend = "QnnCpu.dll"
    elif args.device == "npu":
        backend = "QnnHtp.dll"
    else:
        raise ValueError(f"Device {args.device} is not supported. Supported devices are ['cpu', 'npu']")
    sess = ort.InferenceSession(
        args.model_path, providers=["QNNExecutionProvider"], provider_options=[{"backend_path": backend}]
    )
    # sess = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])

    # load input
    input = np.load(args.input_path, allow_pickle=True)
    input = dict(input.items())

    # output names
    output_names = [o.name for o in sess.get_outputs()]

    # run inference
    if args.type == "accuracy":
        output = sess.run(input_feed=input, output_names=None)
        if len(output_names) == 1:
            output = output[0]
    else:
        latencies = []
        for _ in range(args.warmup_num):
            sess.run([], input)
        for _ in range(args.repeat_test_num):
            t = time.perf_counter()
            sess.run(input_feed=input, output_names=None)
            latencies.append(time.perf_counter() - t)
            time.sleep(args.sleep_num)
        output = np.array(latencies)

    # save output
    np.save(args.output_path, output)


if __name__ == "__main__":
    main()
