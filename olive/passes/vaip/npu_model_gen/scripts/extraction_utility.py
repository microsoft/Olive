##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

# to support the | operator used in the extract_subgraph function arguments.
from __future__ import annotations

import argparse
import onnx

# from dd_helper import onnx_tool
from scripts.helper import *
import os

"""
Included function helps extract a specified sub-graph given an input graph. The cut-points for this subgraph need to be specified in the input/output names.
This helps alleviate the problem of not having value_info's to specify at particular points in the graph. Done by first generating a shape inferred graph
of the original graph given as input, and then performing model extraction from the shape inferred graph to get the desired output.

----FORMAT FOR ARGUMENT----
->  Pass your input/output model path in --input_model and --output_model respectively.
->  The input names and output names can be passed in the following format:
        --input_names input1 input2 input3 --output_names output1 output2 output3
->  If you would like to run model checker on the extracted model, then just pass --check_model
"""


def extract_subgraph(
    input_model: str | os.PathLike,
    shape_infer_model: str | os.PathLike,
    output_model: str | os.PathLike,
    input_names: list[str],
    output_names: list[str],
    check_model: bool,
) -> None:

    m, g = loadmodel(input_model)
    # breakpoint()
    m.graph.shape_infer()
    # m.save_model(shape_infer_model, shape_only=True)
    onnx.save_model(
        m.mproto,
        shape_infer_model,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="temp_model.data",
        size_threshold=1024,
        convert_attribute=False,
    )

    # use the generated shape_inference model to extract the required sub-graph
    onnx.utils.extract_model(
        shape_infer_model, output_model, input_names, output_names, False
    )
    # model = onnx.load(output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="read the comments given in top of the file")

    # defining the template for the argument
    parser.add_argument(
        "--input_model", metavar="input_model", type=str, help="path of input model"
    )
    parser.add_argument(
        "--shape_inferred_model",
        metavar="shape_inferred_model",
        type=str,
        help="path to shape inferred model (output)",
    )
    parser.add_argument(
        "--output_model",
        metavar="output_model",
        type=str,
        help="path to desired sub-graph (output)",
    )
    parser.add_argument(
        "--input_names",
        metavar="input_names",
        nargs="+",
        help="cutoffs to start sub-graph eg: input1 input2 input3",
    )
    parser.add_argument(
        "--output_names",
        metavar="output_names",
        nargs="+",
        help="cutoffs to end sub-graph eg: output1 output2 output3",
    )
    parser.add_argument(
        "--check_model",
        action="store_true",
        help="passing --check_model will set the boolean value to be true",
    )
    args = parser.parse_args()

    # check whether separate shape_inferred_graph is needed
    if args.shape_inferred_model:
        extract_subgraph(
            args.input_model,
            args.shape_inferred_model,
            args.output_model,
            args.input_names,
            args.output_names,
            args.check_model,
        )
    else:
        extract_subgraph(
            args.input_model,
            args.output_model,
            args.output_model,
            args.input_names,
            args.output_names,
            args.check_model,
        )
