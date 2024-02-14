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
    # Map program ids `pid` to the block of Y it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    # -----------------------------------------------------------
    # Iterate to compute a block of the Y matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    # will keep y as fp32 since many triton ops only support fp32/fp64
    y = accumulator

    # -----------------------------------------------------------
    # Indices for the output matrix
    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_idxs = offs_ym[:, None] * N + offs_yn[None, :]
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)

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
    "*{dtype}, *{dtype}, {ptr_dtypes}*{dtype}, i32, i32, i32, {numel_dtypes}{attr_dtypes}{{BLOCK_SIZE_M}},"
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
