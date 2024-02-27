# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add the common directory to the path
ort_inference_utils_parent = Path(__file__).resolve().parents[2] / "common"
sys.path.append(str(ort_inference_utils_parent))

# pylint: disable=wrong-import-position
from ort_inference import get_ort_inference_session, run_inference, time_inference  # noqa: E402


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument(
        "--config_path", type=Path, required=True, help="Path to configuration for the inference to be run"
    )
    parser.add_argument("--model_path", type=Path, required=True, help="Path to onnx model")
    parser.add_argument("--input_dir", type=Path, required=True, help="Path to input directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to output directory")

    return parser.parse_args(raw_args)


def load_batch(input_dir: Path, batch_file: str) -> Dict:
    return dict(np.load(input_dir / batch_file, allow_pickle=True).items())


def main(raw_args=None):
    args = get_args(raw_args)

    # load inference setting
    with args.config_path.open() as f:
        config = json.load(f)

    # create session
    session = get_ort_inference_session(
        args.model_path, config["inference_settings"], config.get("use_ort_extensions", False)
    )

    # mode for the run
    mode = config["mode"]
    # io binding related settings
    io_bind = config.get("io_bind", False)
    device = config.get("device", "cpu")
    shared_kv_buffer = config.get("shared_kv_buffer", False)

    # run inference
    if mode == "inference":

        class DataGenerator:
            def __iter__(self):
                for i in range(config["num_batches"]):
                    yield load_batch(args.input_dir, f"input_{i}.npz"), None

        def post_run(result, idx, input_data, labels):
            np.save(args.output_dir / f"output_{idx}.npy", result)

        run_inference(
            session,
            DataGenerator(),
            post_run=post_run,
            io_bind=io_bind,
            device=device,
            shared_kv_buffer=shared_kv_buffer,
        )
    else:
        warmup_num = config.get("warmup_num")
        repeat_test_num = config.get("repeat_test_num")
        sleep_num = config.get("sleep_num")

        input_data = load_batch(args.input_dir, "input.npz")
        latencies = time_inference(
            session,
            input_data,
            num_runs=warmup_num + repeat_test_num,
            sleep_num=sleep_num,
            io_bind=io_bind,
            device=device,
            shared_kv_buffer=shared_kv_buffer,
        )[warmup_num:]
        np.save(args.output_dir / "output.npy", np.array(latencies))


if __name__ == "__main__":
    main()
