// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/float16.h"
#include "cuda_ops.h"

namespace Cuda {

template <typename T>
void MatMulBNBKernel<T>::Compute(OrtKernelContext* context) {
    // pass
}

void RegisterOps(Ort::CustomOpDomain& domain) {
    static const MatMulBNB<float> c_MatMulBNB_float;
    static const MatMulBNB<onnxruntime::MLFloat16> c_MatMulBNB_float16;

    domain.Add(&c_MatMulBNB_float);
    domain.Add(&c_MatMulBNB_float16);
}

}  // namespace Cuda
