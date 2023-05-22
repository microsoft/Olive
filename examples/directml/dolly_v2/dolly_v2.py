# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
from pathlib import Path
from packaging import version

import os
import onnxruntime as ort

from olive.workflows import run as olive_run


def optimize(model_name: str, optimized_model_dir: Path):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing dolly
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        exit(1)

    ort.set_default_logger_severity(4)
    script_dir = Path(__file__).resolve().parent

    model_info = dict()

    # Optimize the model with Olive
    print(f"\nOptimizing {model_name}")

    olive_config = None
    with open(script_dir / "config_dolly_v2.json", "r") as fin:
        olive_config = json.load(fin)

    olive_run(olive_config)

    # TODO: rename the 0 prefix in the path when the hardware accelerator feature is implemented.
    footprints_file_path = Path(__file__).resolve().parent / "footprints/dolly_v2_0_footprints.json"
    with footprints_file_path.open("r") as footprint_file:
        footprints = json.load(footprint_file)
        conversion_footprint = None
        merger_footprint = None
        for _, footprint in footprints.items():
            if footprint["from_pass"] == "OptimumConversion":
                conversion_footprint = footprint
            elif footprint["from_pass"] == "OptimumMerging":
                merger_footprint = footprint

        assert conversion_footprint and merger_footprint

        unoptimized_config = conversion_footprint["model_config"]["config"]
        optimized_config = merger_footprint["model_config"]["config"]

        model_info = {
            "unoptimized": {
                "path": os.path.dirname(unoptimized_config["model_components"][0]["config"]["model_path"]),
            },
            "optimized": {
                "path": Path(optimized_config["model_path"]),
            },
        }

        print(f"Unoptimized Model : {model_info['unoptimized']['path']}")
        print(f"Optimized Model   : {model_info['optimized']['path']}")

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    shutil.copytree(model_info['unoptimized']['path'], optimized_model_dir, ignore=shutil.ignore_patterns("*.onnx", "*.onnx_data"))

    merged_model_path = str(model_info["optimized"]["path"])
    merged_weights_path = merged_model_path + ".data"

    merged_model_name = os.path.basename(merged_model_path)
    merged_weights_name = merged_model_name + ".data"

    print(f"Copying the optimized model to {optimized_model_dir}")
    shutil.copyfile(merged_model_path, optimized_model_dir / merged_model_name)
    shutil.copyfile(merged_weights_path, optimized_model_dir / merged_weights_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--model", default="databricks/dolly-v2-7b", type=str)
    parser.add_argument("--prompt", default="Explain to me the difference between nuclear fission and fusion.", type=str)
    parser.add_argument("--max_new_tokens", default=64, type=int, help="Maximum number of tokens that the model will generate")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = script_dir / "models" / "optimized" / args.model

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    if args.optimize:
        shutil.rmtree(script_dir / "footprints", ignore_errors=True)
        shutil.rmtree(optimized_model_dir, ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        optimize(args.model, optimized_model_dir)

    if not args.optimize:
        print("This example doesn't support inference yet")
