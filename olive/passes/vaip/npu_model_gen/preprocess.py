##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import os
import sys
import subprocess
from pathlib import Path
from olive.passes.vaip.npu_model_gen.scripts.add_cast_nodes import cast_main, vi_map
from olive.passes.vaip.npu_model_gen.scripts.llm_fusion import fuse_main
import argparse
from olive.passes.vaip.npu_model_gen.scripts.add_pack_consts import pack_main


def process_model(args):
    if isinstance(args, dict):
        input_model = args.get("input_model")
        output_model = args.get("output_model")
        custom_ops = args.get("custom_ops", False)
        fuse = args.get("fuse", False)
        fuse_SSMLP = args.get("fuse_SSMLP", False)
        fuse_GQO = args.get("fuse_GQO", False)
        packed_const = args.get("packed_const", False)
    else:
        input_model = args.input_model
        output_model = args.output_model
        custom_ops = args.custom_ops
        fuse = args.fuse
        fuse_SSMLP = args.fuse_SSMLP
        fuse_GQO = args.fuse_GQO
        packed_const = args.packed_const
    fusion_happening =fuse or  fuse_SSMLP or fuse_GQO

    # Apply casting transformations
    cast_main(
        input_model,
        output_model,
        custom_ops,
        fusion_happening,
        packed_const,
    )
    print("Cast Model generated")
    # Apply constant packing if requested
    if packed_const:
        pack_main(output_model, output_model, fusion_happening)
        print("Packed model generated")

    # Apply fusion transformations
    if fuse:
        fuse_main(output_model, output_model, "SSMLP", "GQO")
    if fuse_SSMLP:
        fuse_main(output_model, output_model, "SSMLP", None)
        print("SSMLP node fused")
    if fuse_GQO:
        fuse_main(output_model, output_model, "GQO", None)

    # Cleanup temporary files
    output_dir = Path(output_model).parent
    cast_data = output_dir / "cast_model.data"
    pack_data = output_dir / "pack_model.data"
    
    if cast_data.exists():
        if fusion_happening or packed_const:
            cast_data.unlink()
    if pack_data.exists() and fusion_happening:
        pack_data.unlink()


def main():
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
    print("Generating model for:", args.input_model)
    process_model(args)


if __name__ == "__main__":
    main()