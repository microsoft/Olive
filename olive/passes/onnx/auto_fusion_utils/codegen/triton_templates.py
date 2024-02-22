# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

ELEMENTWISE_TEMPLATE = """
import triton
import triton.language as tl

# Fusion:
{network}
@triton.jit
def {kernel_name}(
    # input pointers
    {ptr_params}
    # output pointer
    y_ptr,
    # numel of output tensor
    y_numel,
    # shapes of symbolic dimensions
    {shape_params}
    # TODO(jambayk): add attributes once supported
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    \"\"\"Kernel for computing the elementwise operation Y = op(A) or Y = op(A, B).

    A and B are tensors that are multi-directionally broadcastable.
    The output tensor Y has the same shape as the broadcasted A and B.
    \"\"\"
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of Y it should compute.
    pid = tl.program_id(axis=0).to(tl.int64)
    y_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    y_mask = y_idx < y_numel

    # -----------------------------------------------------------
    # Indexing for used dimensions
    {dim_indices}

    # -----------------------------------------------------------
    # Perform operations
    {compute_code}

    # -----------------------------------------------------------
    # Write back the output tensor Y with masks.
    tl.store(y_ptr + y_idx, y, mask=y_mask)
"""

ELEMENTWISE_SIGNATURE = "{ptr_dtypes}*{dtype}, i32, {numel_dtypes}{{BLOCK_SIZE}}"

ELEMENTWISE_GRID = "((y_numel + {BLOCK_SIZE} - 1) / {BLOCK_SIZE}), 1, 1"

ELEMENTWISE_LOAD = """{i_name}_idx = {idx}
    {i_name} = tl.load({i_name}_ptr + {i_name}_idx, mask=y_mask, eviction_policy='evict_last')"""

ELEMENTWISE_LOAD_CONSTANT = """{i_name} = tl.load({i_name}_ptr, eviction_policy='evict_last')"""
