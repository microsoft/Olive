# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict

import olive.passes.onnx.auto_fusion_utils.codegen.triton_templates as templates
from olive.passes.onnx.auto_fusion_utils.codegen.io import KernelIO
from olive.passes.onnx.auto_fusion_utils.codegen.ops import get_op_info
from olive.passes.onnx.auto_fusion_utils.utils import KERNEL_OUTPUT, create_triton_kernel_name, join_params


def create_kernel(kernel_info) -> Dict:
    """Create the kernel for the fused op."""
    # TODO(jambay): Handle matmul fusions again
    kernel_io = KernelIO.from_kernel_info(kernel_info)

    external_inputs = []
    code = []
    for op, inputs in kernel_info["network"]:
        op_inputs = [name if name != KERNEL_OUTPUT else "y" for name in inputs]

        loads = []
        compute_inputs = {}
        for idx, name in enumerate(op_inputs):
            compute_inputs[f"in{idx}"] = name
            if isinstance(name, (int, float)) or name == "y":
                continue
            if name not in external_inputs:
                external_inputs.append(name)

            input_idx = kernel_io.get_input_idx(name)
            if input_idx == "0":
                loads.append(templates.ELEMENTWISE_LOAD_CONSTANT.format(input_name=name))
            else:
                loads.append(templates.ELEMENTWISE_LOAD.format(input_name=name, idx=input_idx))
        compute = f"y = {get_op_info(op).triton_template.format(**compute_inputs)}"
        code.append(join_params([f"# {op}", *loads, compute], joiner="\n    ", end="\n"))

    kernel_name = create_triton_kernel_name(kernel_info)
    kernel_code = templates.ELEMENTWISE_TEMPLATE.format(
        kernel_name=kernel_name,
        network="# TODO(jambayk): add network",
        ptr_params=join_params([f"{name}_ptr" for name in external_inputs]),
        shape_params=join_params(kernel_io.get_symbolic_dims()),
        dim_indices=join_params(kernel_io.get_dim_indices(), joiner="\n    ", end=""),
        compute_code=join_params(code, joiner="\n    ", end=""),
    )
    # only fp32 for now
    signature = templates.ELEMENTWISE_SIGNATURE.format(
        ptr_dtypes=join_params(["*fp32"] * len(external_inputs), joiner=", ", end=", ", default=""),
        dtype="fp32",
        shape_dtypes=join_params(["i32"] * len(kernel_io.get_symbolic_dims()), joiner=", ", end=", ", default=""),
    )
    return {
        "kernel_name": kernel_name,
        "kernel_code": kernel_code,
        "signature": signature,
        "grid": templates.ELEMENTWISE_GRID,
    }
