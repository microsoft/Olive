# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict, List, Tuple

import olive.passes.onnx.auto_fusion_utils.codegen.ort_templates as templates
from olive.passes.onnx.auto_fusion_utils.codegen.ops import get_num_op_inputs, get_op_info
from olive.passes.onnx.auto_fusion_utils.utils import (
    CPP_DTYPE_MAP,
    create_custom_op_name,
    create_triton_kernel_name,
    join_params,
)


def get_op_args(op: str, op_idx: int, cpp_dtype: str) -> Dict:
    """Create the op related template arguments to use in the fused op template.

    Currently, we only support elementwise ops.
    """
    # number of inputs for this op
    # first input is always the output of the previous op
    num_inputs = get_num_op_inputs(op)
    op_info = get_op_info(op)
    # unique name for this op
    unique_op_name = f"{op}_{op_idx}".lower()

    # args to be used in the full template
    template_args = {
        "input_param": None,
        "attr_params": [],
        "input_shape_validation": None,
        "input_arg": None,
        "numel_arg": None,
        "attr_args": [],
    }

    # create arg for second input if exists
    if num_inputs == 2:
        in1 = f"{unique_op_name}_in1"
        template_args["input_param"] = templates.INPUT_PARAM.format(cpp_dtype=cpp_dtype, input_name=in1)
        template_args["input_shape_validation"] = templates.INPUT_SHAPE_VALIDATION.format(input_name=in1)
        template_args["input_arg"] = templates.INPUT_ARG.format(input_name=in1)
        template_args["numel_arg"] = templates.NUMEL_ARG.format(input_name=in1)

    # create args for attributes if any
    for attr_name, attr_dtype in op_info.attributes or []:
        attr_arg = f"{unique_op_name}_{attr_name}"
        template_args["attr_params"].append(
            templates.ATTR_PARAM.format(attr_cpp_dtype=CPP_DTYPE_MAP[attr_dtype], attr_name=attr_arg)
        )
        template_args["attr_args"].append(attr_arg)

    return template_args


def create_custom_op(base_op: str, fused_ops: List[str], dtype: str, constants: Dict) -> Tuple[str, str]:
    """Create the custom op for the fused op.

    Returns the custom op name and code.
    """
    op_names = [base_op, *fused_ops]
    def_template = templates.MATMUL_TEMPLATE if base_op == "MatMul" else templates.ELEMENTWISE_TEMPLATE
    default_comment = "// Not used"
    kernel_name = create_triton_kernel_name(op_names, dtype)
    custom_op_name = create_custom_op_name(op_names, dtype)
    cpp_dtype = CPP_DTYPE_MAP[dtype]

    # create args for fused ops
    input_params = []
    attr_params = []
    input_shape_validations = []
    input_args = []
    numel_args = []
    attr_args = []
    for op_idx, op in enumerate(fused_ops if base_op == "MatMul" else op_names):
        template_args = get_op_args(op, op_idx, cpp_dtype)
        if template_args["input_param"]:
            input_params.append(template_args["input_param"])
            input_shape_validations.append(template_args["input_shape_validation"])
            input_args.append(template_args["input_arg"])
            numel_args.append(template_args["numel_arg"])
        attr_params.extend(template_args["attr_params"] or [])
        attr_args.extend(template_args["attr_args"] or [])
    template_args = {
        "cpp_dtype": cpp_dtype,
        "custom_op_name": custom_op_name,
        "kernel_name": kernel_name,
        "input_params": join_params(input_params, default=default_comment),
        "attr_params": join_params(attr_params, default=default_comment),
        "input_shape_validation": join_params(input_shape_validations, joiner="\n  ", end="", default=default_comment),
        "input_args": join_params(input_args, default=default_comment),
        "numel_args": join_params(numel_args, default=default_comment),
        "attr_args": join_params(attr_args, default=default_comment),
    }
    for constant_group in constants.values():
        template_args.update(constant_group)

    return {
        "op_name": custom_op_name,
        "op_def": def_template.format(**template_args),
        "kernel_include": templates.CUSTOM_KERNEL_INCLUDE.format(kernel_name=kernel_name),
        "op_registration": templates.CUSTOM_OP_REGISTRATION.format(custom_op_name=custom_op_name),
    }


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
