# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import pickle
import sys
from pathlib import Path

ort_inference_utils_parent = Path(__file__).resolve().parent.parent.parent / "common"
sys.path.append(str(ort_inference_utils_parent))

# pylint: disable=wrong-import-position
from ort_inference import get_ort_inference_session  # noqa: E402


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Check if execution provider is valid")

    parser.add_argument("--model_path", type=str, required=True, help="Path to onnx model")
    parser.add_argument("--ep", type=str, required=True, help="Execution provider to check")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output to")

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # inference settings
    inference_settings = {"execution_provider": args.ep}

    # create inference session
    try:
        get_ort_inference_session(args.model_path, inference_settings)
        with Path(args.output_path).open("wb") as f:
            pickle.dump({"valid": True}, f)
    except Exception as e:
        with Path(args.output_path).open("wb") as f:
            pickle.dump({"valid": False, "error": str(e)}, f)


if __name__ == "__main__":
    main()
