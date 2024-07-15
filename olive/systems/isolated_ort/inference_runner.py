# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# NOTE: Only onnxruntime and its dependencies can be imported in this file!!!
# Import them lazily since onnxruntime is not a required dependency for Olive.
# Import in TYPE_CHECKING block for type hinting is fine.
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
from ort_inference import OrtInferenceSession, get_ort_inference_session  # noqa: E402


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Onnx model inference")

    parser.add_argument(
        "--config_path", type=Path, required=True, help="Path to configuration for the inference to be run"
    )
    parser.add_argument("--model_path", type=Path, required=True, help="Path to onnx model")
    parser.add_argument("--external_initializers_path", type=Path, help="Path to external initializers")
    parser.add_argument("--constant_inputs_path", type=Path, help="Path to constant inputs")
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
        args.model_path,
        config["inference_settings"],
        config.get("use_ort_extensions", False),
        external_initializers=np.load(args.external_initializers_path) if args.external_initializers_path else None,
    )

    # load constant inputs
    constant_inputs = None
    if args.constant_inputs_path:
        constant_inputs = np.load(args.constant_inputs_path)

    # get first batch
    input_feed = load_batch(args.input_dir, "input_0.npz" if config["mode"] == "inference" else "input.npz")
    session_wrapper = OrtInferenceSession(
        session,
        io_bind=config.get("io_bind", False),
        device=config.get("device", "cpu"),
        shared_kv_buffer=config.get("shared_kv_buffer", False),
        use_fp16=config.get("use_fp16", False),
        input_feed=input_feed,
        constant_inputs=constant_inputs,
    )

    # run inference
    if config["mode"] == "inference":
        for i in range(config["num_batches"]):
            input_feed = load_batch(args.input_dir, f"input_{i}.npz")
            result = session_wrapper.run(None, input_feed)
            np.save(args.output_dir / f"output_{i}.npy", result)
    else:
        warmup_num = config["warmup_num"]
        repeat_test_num = config["repeat_test_num"]
        sleep_num = config["sleep_num"]

        input_feed = load_batch(args.input_dir, "input.npz")
        latencies = session_wrapper.time_run(
            input_feed, num_runs=repeat_test_num, num_warmup=warmup_num, sleep_time=sleep_num
        )
        np.save(args.output_dir / "output.npy", np.array(latencies))


if __name__ == "__main__":
    main()
