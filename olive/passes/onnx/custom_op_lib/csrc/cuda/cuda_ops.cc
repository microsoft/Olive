// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <iostream>
#include "onnxruntime_lite_custom_op.h"

#include "cuda_ops.h"

// using namespace Ort::Custom;

namespace Cuda {

template <typename T>
void MatMulBnb4Kernel<T>::Compute(OrtKernelContext* context) {
    const Ort::Custom::Tensor<T>& A = Ort::Custom::Tensor<T>(context, 0, 1);
    const Ort::Custom::Tensor<u_int8_t>& B_quant = Ort::Custom::Tensor<u_int8_t>(context, 1, 1);
    const Ort::Custom::Tensor<int64_t>& B_shape = Ort::Custom::Tensor<int64_t>(context, 3, 1);

    const float_t* absmax_value;
    if (double_quant_) {
        const Ort::Custom::Tensor<u_int8_t>& absmax_int8 = Ort::Custom::Tensor<u_int8_t>(context, 2, 1);
        const Ort::Custom::Tensor<float_t>& offset = Ort::Custom::Tensor<float_t>(context, 4, 1);
        const Ort::Custom::Tensor<float_t>& nested_absmax = Ort::Custom::Tensor<float_t>(context, 5, 1);
        const Ort::Custom::Tensor<float_t>& nested_code = Ort::Custom::Tensor<float_t>(context, 6, 1);

        // TODO(jambayk): dequantize absmax_int8
    } else {
        const Ort::Custom::Tensor<float_t>& absmax_float = Ort::Custom::Tensor<float_t>(context, 2, 1);
        absmax_value = absmax_float.Data();
    }

}

void RegisterOps(Ort::CustomOpDomain& domain) {
    static const MatMulBnb4<float_t> c_MatMulBnb4_float;
    static const MatMulBnb4<Ort::Float16_t> c_MatMulBnb4_float16;

    domain.Add(&c_MatMulBnb4_float);
    domain.Add(&c_MatMulBnb4_float16);
}

}  // namespace Cuda
