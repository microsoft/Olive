# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import importlib.util
import pickle
import time
from pathlib import Path
from typing import Union

import numpy as np


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument("--type", type=str, required=True, choices=["accuracy", "latency"], help="Type of metric")
    parser.add_argument("--model_path", type=str, required=True, help="Path to onnx model")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--inference_settings_path", type=str, required=True, help="Path to inference settings file")

    # accuracy args
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to run. Only for accuracy metric")

    # latency args
    parser.add_argument(
        "--warmup_num", type=int, default=10, help="Number of warmup iterations. Only for latency metric"
    )
    parser.add_argument("--repeat_test_num", type=int, default=20, help="Number of iterations. Only for latency metric")
    parser.add_argument("--sleep_num", type=int, default=0, help="Number of sleep iterations. Only for latency metric")
    parser.add_argument("--io_bind", type=bool, default=False, help="Use IO binding. Only for latency metric")
    parser.add_argument("--device", type=str, default="cpu", help="Device to io bind on")

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)

    ort_inference_utils_path = Path(__file__).resolve().parent.parent.parent / "common" / "ort_inference.py"
    ort_inference_utils = import_module_from_file(ort_inference_utils_path)
    get_ort_inference_session = getattr(ort_inference_utils, "get_ort_inference_session")

    # load inference setting
    inference_settings = pickle.load(open(args.inference_settings_path, "rb"))

    # create session
    sess = get_ort_inference_session(args.model_path, inference_settings)

    if args.type == "accuracy":
        for i in range(args.num_batches):
            # load input
            input_dict = np.load(args.input_dir / f"input_{i}.npz", allow_pickle=True)
            input_dict = dict(input_dict.items())
            output = sess.run(input_feed=input_dict, output_names=None)

            # save output
            np.save(args.output_dir / f"output_{i}.npy", output)
    else:
        # load input
        input_dict = np.load(args.input_dir / "input.npz", allow_pickle=True)
        input_dict = dict(input_dict.items())

        if args.io_bind:
            io_bind_op = sess.io_binding()
            io_bind_device = "cuda" if args.device == "gpu" else "cpu"
            for k, v in input_dict.items():
                io_bind_op.bind_cpu_input(k, v)
            for item in sess.get_outputs():
                io_bind_op.bind_output(item.name, io_bind_device)

        for _ in range(args.warmup_num):
            if args.io_bind:
                sess.run_with_iobinding(io_bind_op)
            else:
                sess.run(input_feed=input_dict, output_names=None)

        latencies = []
        for _ in range(args.repeat_test_num):
            if args.io_bind:
                t = time.perf_counter()
                sess.run_with_iobinding(io_bind_op)
                latencies.append(time.perf_counter() - t)
            else:
                t = time.perf_counter()
                sess.run(input_feed=input_dict, output_names=None)
                latencies.append(time.perf_counter() - t)
            time.sleep(args.sleep_num)

        output = np.array(latencies)
        # save output
        np.save(args.output_dir / "output.npy", output)


def import_module_from_file(module_path: Union[Path, str], module_name: str = None):
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise ValueError(f"{module_path} doesn't exist")

    if module_name is None:
        if module_path.is_dir():
            module_name = module_path.name
            module_path = module_path / "__init__.py"
        elif module_path.name == "__init__.py":
            module_name = module_path.parent.name
        else:
            module_name = module_path.stem

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    new_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_module)
    return new_module


if __name__ == "__main__":
    main()
