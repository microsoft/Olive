// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "core/framework/float16.h"
#include "cuda_ops.h"

namespace Cuda {

template <typename T>
void MatMulBNBKernel<T>::Compute(OrtKernelContext* context) {
    // pass
}

}  // namespace Cuda
