# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import os
import shutil
from pathlib import Path

import config
import onnxruntime as ort
from packaging import version

from olive.model import CompositeOnnxModel, ONNXModel
from olive.workflows import run as olive_run


def optimize(model_name: str, optimized_model_dir: Path):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing dolly
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        exit(1)

    ort.set_default_logger_severity(4)
    script_dir = Path(__file__).resolve().parent

    model_info = {}

    # Optimize the model with Olive
    print(f"\nOptimizing {model_name}")

    olive_config = None
    with (script_dir / "config_dolly_v2.json").open() as fin:
        olive_config = json.load(fin)

    olive_config["input_model"]["config"]["model_path"] = model_name
    olive_config["passes"]["optimize"]["config"]["hidden_size"] = config.hidden_size
    olive_run(olive_config)

    # TODO(PatriceVignola): rename the 0 prefix in the path when the hardware accelerator feature is implemented.
    footprints_file_path = Path(__file__).resolve().parent / "footprints/dolly_v2_gpu-dml_footprints.json"
    with footprints_file_path.open("r") as footprint_file:
        footprints = json.load(footprint_file)
        conversion_footprint = None
        merger_footprint = None
        for footprint in footprints.values():
            if footprint["from_pass"] == "OptimumConversion":
                conversion_footprint = footprint
            elif footprint["from_pass"] == "OptimumMerging":
                merger_footprint = footprint

        assert conversion_footprint and merger_footprint

        unopimized_olive_model = CompositeOnnxModel(**conversion_footprint["model_config"]["config"])
        optimized_olive_model = ONNXModel(**merger_footprint["model_config"]["config"])

        model_info = {
            "unoptimized": {
                "path": Path(unopimized_olive_model.get_model_component(0).model_path).parent,
            },
            "optimized": {
                "path": Path(optimized_olive_model.model_path),
            },
        }

        print(f"Unoptimized Model : {model_info['unoptimized']['path']}")
        print(f"Optimized Model   : {model_info['optimized']['path']}")

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    shutil.copytree(
        model_info["unoptimized"]["path"], optimized_model_dir, ignore=shutil.ignore_patterns("*.onnx", "*.onnx_data")
    )

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
    parser.add_argument(
        "--prompt", default="Explain to me the difference between nuclear fission and fusion.", type=str
    )
    parser.add_argument(
        "--max_new_tokens", default=64, type=int, help="Maximum number of tokens that the model will generate"
    )
    args = parser.parse_args()

    model_to_hidden_size = {
        "databricks/dolly-v2-3b": 2560,
        "databricks/dolly-v2-7b": 4096,
    }

    if args.model not in list(model_to_hidden_size.keys()):
        print(
            f"WARNING: {args.model} is not an officially supported model for this example and may not work as expected."
        )

    config.hidden_size = model_to_hidden_size.get(args.model, 2560)

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
