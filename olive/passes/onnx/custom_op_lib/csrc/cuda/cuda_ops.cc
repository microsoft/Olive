// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "cuda_ops.h"

namespace Cuda {

template <typename T>
void MatMulBnb4Kernel<T>::Compute(OrtKernelContext* context) {
    // pass
}

void RegisterOps(Ort::CustomOpDomain& domain) {
    static const MatMulBnb4<float> c_MatMulBnb4_float;
    static const MatMulBnb4<Ort::Float16_t> c_MatMulBnb4_float16;

    domain.Add(&c_MatMulBnb4_float);
    domain.Add(&c_MatMulBnb4_float16);
}

}  // namespace Cuda
