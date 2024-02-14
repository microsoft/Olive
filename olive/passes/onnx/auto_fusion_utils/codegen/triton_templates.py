# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

ELEMENTWISE_TEMPLATE = """
import triton
import triton.language as tl

# Fused operations: {fused_ops_str}
@triton.jit
def {kernel_name}(
    # pointers to base input tensor
    a_ptr,
    # pointer to other tensors
    {ptr_params}
    # pointer to output tensor
    y_ptr,
    # number of elements for base input tensor
    a_numel,
    # number of elements for other tensors
    {numel_params}
    # attributes for ops
    {attr_params}
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    \"\"\"Kernel for computing the elementwise operation Y = op(A) or Y = op(A, B).

    A has shape (a_numel,) and B has shape (b_numel,) where a_numel % b_numel == 0.
    The output Y has shape (a_numel,).
    Elementwise operation can be fused with other elementwise operations.
    \"\"\"
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of Y it should compute.
    pid = tl.program_id(axis=0).to(tl.int64)
    y_idxs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    y_mask = y_idxs < a_numel

    # -----------------------------------------------------------
    # Load first input tensor
    y = tl.load(a_ptr + y_idxs, mask=y_mask)
    # will use fp32 since many triton ops only support fp32/fp64
    # TODO: investigate if we can use fp16
    y = y.to(tl.float32)

    # -----------------------------------------------------------
    # Perform operations
    {fused_code}

    # -----------------------------------------------------------
    # Write back the output tensor Y with masks.
    y = y.to({tl_dtype})
    y_ptrs = y_ptr + y_idxs
    tl.store(y_ptrs, y, mask=y_mask)
"""

ELEMENTWISE_SIGNATURE = "*{dtype}, {ptr_dtypes}*{dtype}, i32, {numel_dtypes}{attr_dtypes}{{BLOCK_SIZE}}"

ELEMENTWISE_GRID = "((a_numel + {BLOCK_SIZE} - 1) / {BLOCK_SIZE}), 1, 1"

MATMUL_TEMPLATE = """
import triton
import triton.language as tl

# Fused operations: {fused_ops_str}
@triton.jit
def {kernel_name}(
    # pointers to matrices
    a_ptr,
    b_ptr,
    # pointer to other tensors for fused operations
    {ptr_params}
    # pointer to output tensor
    y_ptr,
    # matrix dimensions
    M,
    N,
    K,
    EVEN_K,
    # number of elements for other tensors
    {numel_params}
    # attributes for fused operations
    {attr_params}
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    \"\"\"Kernel for computing the matmul Y = A x B.

    A has shape (M, K), B has shape (K, N) and Y has shape (M, N).
    Matmul can be fused with elementwise operations such as bias addition, activation, etc.
    \"\"\"
    # -----------------------------------------------------------
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # re-order program ID for better L2 performance
    width = GROUP_SIZE_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # do matrix multiplication
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_SIZE_N), BLOCK_SIZE_N)
    rk = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # pointers
    a_ptr = a_ptr + (ram[:, None] * K + rk[None, :])
    b_ptr = b_ptr + (rk[:, None] * N + rbn[None, :])
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            _0 = tl.zeros((1, 1), dtype=tl.float32)
            a = tl.load(a_ptr, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(b_ptr, mask=rk[:, None] < k_remaining, other=_0)

        acc += tl.dot(a, b)
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K * N

    # will keep y as fp32 since many triton ops only support fp32/fp64
    y = acc

    # -----------------------------------------------------------
    # Indices for the output matrix
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_idxs = rm[:, None] * N + rn[None, :]
    y_mask = (rm < M)[:, None] & (rn < N)[None, :]

    # -----------------------------------------------------------
    # Fusion with other operations
    {fused_code}

    # -----------------------------------------------------------
    # Write back the block of the output matrix Y with masks.
    y = y.to({tl_dtype})
    y_ptrs = y_ptr + y_idxs
    tl.store(y_ptrs, y, mask=y_mask)
"""

MATMUL_SIGNATURE = (
    "*{dtype}, *{dtype}, {ptr_dtypes}*{dtype}, i32, i32, i32, i32, {numel_dtypes}{attr_dtypes}{{BLOCK_SIZE_M}},"
    " {{BLOCK_SIZE_N}}, {{BLOCK_SIZE_K}}, {{GROUP_SIZE_M}}"
)

MATMUL_GRID = "((M + {BLOCK_SIZE_M} - 1) / {BLOCK_SIZE_M} * (N + {BLOCK_SIZE_N} - 1) / {BLOCK_SIZE_N}), 1, 1"

FUSED_OP_TWO_INPUT_TEMPLATE = """
    # load the second input tensor
    {in1}_idxs = y_idxs % {in1_numel}
    {in1}_ptrs = {in1_ptr} + {in1}_idxs
    {in1} = tl.load({in1}_ptrs, mask=y_mask, eviction_policy='evict_last')
    {in1} = {in1}.to(tl.float32)
    # perform the operation
    {op_code}"""
