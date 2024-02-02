# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

CUSTOM_OP_SKELETON = """
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include <cuda.h>
#include <iostream>

#include "core/providers/cuda/cuda_context.h"
#include "onnxruntime_lite_custom_op.h"

// Include custom kernels headers
{custom_kernel_includes}


using namespace Ort::Custom;

#define CUSTOM_ENFORCE(cond, msg)  \\
  if (!(cond)) {{                   \\
    throw std::runtime_error(msg); \\
  }}

void ValidateElementwiseShapes(
    const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape) {{
  // currently, we only support limited one-directional broadcasting
  // 1s can only be prepended to the second input
  // second input withou leading 1s must be a suffix of the first input

  // check that the shapes are compatible
  CUSTOM_ENFORCE(b_shape.size() <= a_shape.size(), "Second input cannot have more dimensions than the first input");

  // check that the trailing dimensions match
  bool leading_ones = true;
  bool mismatch = false;
  for (size_t i = 0; i < b_shape.size(); ++i) {{
    int64_t a_shape_i = a_shape[a_shape.size() - b_shape.size() + i];
    int64_t b_shape_i = b_shape[i];
    // skip leading ones in the second input
    if (leading_ones && b_shape_i == 1) continue;
    // once we see a non-one dimension, we are no longer skipping 1s in the second input
    leading_ones = false;
    // check that the dimensions match
    if (a_shape_i != b_shape_i) {{
      mismatch = true;
      break;
    }}
  }}
  CUSTOM_ENFORCE(!mismatch, "Input shapes are not compatible");
}}

namespace OliveTritonFusion {{

// Define Custom Ops
{custom_op_defs}

// Register Custom Ops
void RegisterOps(Ort::CustomOpDomain& domain) {{
  {custom_op_registrations}
}}

}} // namespace OliveTritonFusion
"""

CUSTOM_KERNEL_INCLUDE = '#include "{kernel_name}/{kernel_name}.h"'

CUSTOM_OP_REGISTRATION = """
  // Register {custom_op_name}
  static const std::unique_ptr<OrtLiteCustomOp> c_{custom_op_name}{{
    Ort::Custom::CreateLiteCustomOp("{custom_op_name}", "CUDAExecutionProvider", {custom_op_name})
  }};
  domain.Add(c_{custom_op_name}.get());"""

MATMUL_TEMPLATE = """
void {custom_op_name}(
    const Ort::Custom::CudaContext& cuda_ctx,
    // matmul inputs
    const Ort::Custom::Tensor<{cpp_dtype}>& a,
    const Ort::Custom::Tensor<{cpp_dtype}>& b,
    // fused Op inputs
    {input_params}
    // fused Op attributes
    {attr_params}
    // output
    Ort::Custom::Tensor<float>& y) {{

  // get shape of a: M1 X ... X Mn X K.
  // Ex. batch_size X seq_len X K
  auto a_shape = a.Shape();
  // stack all dimensions except the last one
  int64_t M = std::accumulate(a_shape.begin(), a_shape.end() - 1, 1, std::multiplies<int64_t>());
  // last dimension
  int64_t K = a_shape.back();

  // currently, we will only support 2D tensors for b
  // shape of b: K X N
  auto b_shape = b.Shape();
  CUSTOM_ENFORCE(b_shape.size() == 2, "b must be a 2D tensor");
  CUSTOM_ENFORCE(b_shape[0] == K, "First dimension of b must be equal to last dimension of a");
  int64_t N = b_shape[1];

  // shape of output: M1 X ... X Mn X N
  std::vector<int64_t> y_shape(a_shape.size());
  std::copy(a_shape.begin(), a_shape.end() - 1, y_shape.begin());
  y_shape.back() = N;

  // print the shapes
  // std::cout << "{custom_op_name} shapes: M=" << M << ", N=" << N << ", K=" << K << std::endl;

  // validate shapes of fused inputs
  {input_shape_validation}

  // allocate output tensor
  auto y_raw = y.Allocate(y_shape);

  // call the kernel
  load_{kernel_name}();
  CUresult ret = {kernel_name}(
      cuda_ctx.cuda_stream,
      reinterpret_cast<CUdeviceptr>(a.DataRaw()),
      reinterpret_cast<CUdeviceptr>(b.DataRaw()),
      {input_args}
      reinterpret_cast<CUdeviceptr>(y_raw),
      M, N, K,
      {numel_args}
      {attr_args}
      0);
  CUSTOM_ENFORCE(ret == CUDA_SUCCESS, "{kernel_name}_default failed");
  unload_{kernel_name}();
}}
"""

ELEMENTWISE_TEMPLATE = """
void {custom_op_name}(
    const Ort::Custom::CudaContext& cuda_ctx,
    // base input
    const Ort::Custom::Tensor<{cpp_dtype}>& a,
    // other inputs
    {input_params}
    // attributes
    {attr_params}
    // output
    Ort::Custom::Tensor<{cpp_dtype}>& y) {{

  // output shape is the same as input shape
  // true because we only support limited one-directional broadcasting
  auto y_shape = a.Shape();

  // print the shapes
  // std::cout << "{custom_op_name} shapes: a_numel=" << a.NumberOfElement() << std::endl;

  // validate shapes of other inputs
  {input_shape_validation}

  // allocate output tensor
  auto y_raw = y.Allocate(y_shape);

  // call the kernel
  load_{kernel_name}();
  CUresult ret = {kernel_name}(
      cuda_ctx.cuda_stream,
      reinterpret_cast<CUdeviceptr>(a.DataRaw()),
      {input_args}
      reinterpret_cast<CUdeviceptr>(y_raw),
      a.NumberOfElement(),
      {numel_args}
      {attr_args}
      0);
  CUSTOM_ENFORCE(ret == CUDA_SUCCESS, "{kernel_name} failed");
  unload_{kernel_name}();
}}
"""

INPUT_PARAM = "const Ort::Custom::Tensor<{cpp_dtype}>& {input_name}"

ATTR_PARAM = "{attr_cpp_dtype} {attr_name}"

INPUT_SHAPE_VALIDATION = """ValidateElementwiseShapes(y_shape, {input_name}.Shape());
  // std::cout << "{input_name}_numel=" << {input_name}.NumberOfElement() << std::endl;"""

INPUT_ARG = "reinterpret_cast<CUdeviceptr>({input_name}.DataRaw())"

NUMEL_ARG = "{input_name}.NumberOfElement()"
