##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import os
import sys
import subprocess
from pathlib import Path
from scripts.add_cast_nodes import cast_main, vi_map
from scripts.llm_fusion import fuse_main
import argparse

# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pack_consts"))
# )
from scripts.add_pack_consts import pack_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model paths.")
    parser.add_argument(
        "--input_model", type=str, required=True, help="Path to your input model"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Path to save your cast-fused model",
    )
    parser.add_argument(
        "--custom_ops",
        action="store_true",
        help="Pass this if AMD custom_ops required",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Pass this if fused SSMLP+GQO model is required",
    )
    parser.add_argument(
        "--fuse_SSMLP",
        action="store_true",
        help="Pass this if fused SSMLP model is required",
    )
    parser.add_argument(
        "--fuse_GQO",
        action="store_true",
        help="Pass this if fused GQO model is required",
    )
    parser.add_argument(
        "--packed_const",
        action="store_true",
        help="Pass this if packed constants are required",
    )
    args = parser.parse_args()

    input_model = args.input_model
    output_model = args.output_model

    print("Generating model for: ", input_model)

    fusion_happening = args.fuse or args.fuse_GQO or args.fuse_SSMLP

    # Running the cast script, this always happens
    cast_main(
        input_model, output_model, args.custom_ops, fusion_happening, args.packed_const
    )
    print("Cast Model generated")

    # Running the packed const script, optional
    if args.packed_const:
        pack_main(output_model, output_model, fusion_happening)
        print("Packed model generated")

    # Running the fusion script, optional
    if args.fuse:
        fuse_main(output_model, output_model, "SSMLP", "GQO")
    if args.fuse_SSMLP:
        fuse_main(output_model, output_model, "SSMLP", None)
    if args.fuse_GQO:
        fuse_main(output_model, output_model, "GQO", None)

    # Deleting the temp file

    output_dir = Path(output_model).parent
    file_to_delete_1 = os.path.join(output_dir, "cast_model.data")
    file_to_delete_2 = os.path.join(output_dir, "pack_model.data")
    if os.path.exists(file_to_delete_1):
        if fusion_happening or args.packed_const:
            os.remove(file_to_delete_1)
    if os.path.exists(file_to_delete_2) and fusion_happening:
        os.remove(file_to_delete_2)
