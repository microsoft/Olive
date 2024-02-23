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

ELEMENTWISE_TEMPLATE = """
void {custom_op_name}(
    const Ort::Custom::CudaContext& cuda_ctx,
    // input tensors
    {input_params}
    // output
    Ort::Custom::Tensor<{cpp_dtype}>& y) {{

  // shapes of tensors that will be used to get dimensions
  {shapes}

  // values of symbolic dimensions
  {dims}

  // output shape
  std::vector<int64_t> y_shape {y_shape};
  int64_t y_numel = std::accumulate(y_shape.begin(), y_shape.end(), 1LL, std::multiplies<int64_t>());

  // allocate output tensor
  auto y_raw = y.Allocate(y_shape);

  // call the kernel
  CUresult ret = {kernel_name}(
      cuda_ctx.cuda_stream,
      {input_args}
      reinterpret_cast<CUdeviceptr>(y_raw),
      y_numel,
      {dim_args}
      0);
  CUSTOM_ENFORCE(ret == CUDA_SUCCESS, "{kernel_name} failed");
}}
"""

INPUT_PARAM = "const Ort::Custom::Tensor<{cpp_dtype}>& {input_name}"

INPUT_ARG = "reinterpret_cast<CUdeviceptr>({input_name}.DataRaw())"

SHAPE = "auto {input_name}_shape = {input_name}.Shape();"

DIM = "int64_t {dim_name} = {input_name}_shape[{idx}];"
