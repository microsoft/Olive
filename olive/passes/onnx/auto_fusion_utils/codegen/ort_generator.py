# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict, List

import olive.passes.onnx.auto_fusion_utils.codegen.ort_templates as templates
from olive.passes.onnx.auto_fusion_utils.codegen.io import KernelIO
from olive.passes.onnx.auto_fusion_utils.utils import (
    CPP_DTYPE_MAP,
    KERNEL_OUTPUT,
    create_custom_op_name,
    create_triton_kernel_name,
    join_params,
)


def create_custom_op(kernel_info: Dict) -> Dict:
    """Create custom op for the fused kernel."""
    kernel_io = KernelIO.from_kernel_info(kernel_info)

    default_comment = "// Not used"
    kernel_name = create_triton_kernel_name(kernel_info)
    custom_op_name = create_custom_op_name(kernel_info)
    # TODO(jambayk): Add support for different dtypes again
    cpp_dtype = CPP_DTYPE_MAP["fp32"]

    seen_shapes = set()
    shapes = []
    dims = []
    dim_args = []
    for name in kernel_io.get_symbolic_dims():
        source, idx = kernel_io.get_dim_source(name)
        if source not in seen_shapes:
            seen_shapes.add(source)
            shapes.append(templates.SHAPE.format(input_name=source))
        dims.append(templates.DIM.format(dim_name=name, input_name=source, idx=idx))
        dim_args.append(name)

    seen_inputs = set()
    input_params = []
    input_args = []
    for _, inputs in kernel_info["network"]:
        for name in inputs:
            if isinstance(name, (int, float)) or name == KERNEL_OUTPUT or name in seen_inputs:
                continue

            input_params.append(templates.INPUT_PARAM.format(input_name=name, cpp_dtype=cpp_dtype))
            input_args.append(templates.INPUT_ARG.format(input_name=name))
            seen_inputs.add(name)

    op_def = templates.ELEMENTWISE_TEMPLATE.format(
        cpp_dtype=cpp_dtype,
        custom_op_name=custom_op_name,
        kernel_name=kernel_name,
        input_params=join_params(input_params, default=default_comment),
        shapes=join_params(shapes, joiner="\n  ", end="", default=default_comment),
        dims=join_params(dims, joiner="\n  ", end="", default=default_comment),
        y_shape=f"{{ {', '.join(map(str, kernel_io.output_shape))} }}",
        input_args=join_params(input_args, joiner=",\n      ", default=default_comment),
        dim_args=join_params(dim_args, joiner=",\n      ", default=default_comment),
    )
    return {
        "op_name": custom_op_name,
        "op_def": op_def,
        "kernel_include": templates.CUSTOM_KERNEL_INCLUDE.format(kernel_name=kernel_name),
        "op_registration": templates.CUSTOM_OP_REGISTRATION.format(custom_op_name=custom_op_name),
    }


def join_code(code_list: List[str]) -> str:
    """Join all code snippets into one string."""
    return join_params(code_list, joiner="\n  ", end="", default="// Not used")


def join_custom_ops(custom_ops_info: List[Dict]) -> str:
    """Join all custom ops into one string."""
    kernel_includes = []
    op_defs = []
    op_registrations = []
    for op_info in custom_ops_info:
        kernel_includes.append(op_info["kernel_include"])
        op_defs.append(op_info["op_def"])
        op_registrations.append(op_info["op_registration"])

    return templates.CUSTOM_OP_SKELETON.format(
        custom_kernel_includes="\n".join(kernel_includes),
        custom_op_defs="\n".join(op_defs),
        custom_op_registrations="\n".join(op_registrations),
    )
