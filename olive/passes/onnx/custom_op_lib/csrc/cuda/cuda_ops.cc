// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <iostream>
#include <cuda_runtime.h>

#include "cuda_ops.h"


namespace Cuda {

template <typename T>
void BnbDequantizeKernel<T>::Compute(OrtKernelContext* context) {
    // first input is not used currently
    // keeping it to infer the type of the weight
    // might also help execution order so that the BnbDequantize node has a parent
    // TODO(jambayk): clean this up so that we don't need to pass the first input
    Ort::KernelContext ctx(context);
    auto B_quant = ctx.GetInput(1);
    auto B_shape = ctx.GetInput(3);

    const float_t* absmax_value;
    if (double_quant_) {
        auto absmax_int8 = ctx.GetInput(2);
        auto offset = ctx.GetInput(4);
        auto nested_absmax = ctx.GetInput(5);
        auto nested_code = ctx.GetInput(6);

        // TODO(jambayk): dequantize absmax_int8, move the value to device, return pointer
    } else {
        auto absmax_float = ctx.GetInput(2);
        absmax_value = absmax_float.GetTensorData<float_t>();
    }

    // get shape of output
    size_t B_shape_size = B_shape.GetTensorTypeAndShapeInfo().GetElementCount();
    int64_t* B_shape_local = new int64_t[B_shape_size];
    cudaMemcpyAsync(B_shape_local, B_shape.GetTensorData<int64_t>(), B_shape_size * sizeof(int64_t), cudaMemcpyDeviceToHost,
               reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream()));

    // get output and allocate memory
    // TODO(jambayk): the output of the dequantize function can be different from T
    // it depends on the dtype_ attribute
    // need to find a better way to handle this
    auto B_dequant = ctx.GetOutput(0, B_shape_local, B_shape_size);
    T* B_dequant_data = B_dequant.GetTensorMutableData<T>();
}

void RegisterOps(Ort::CustomOpDomain& domain) {
    static const BnbDequantize<float_t> c_BnbDequantize_float;
    static const BnbDequantize<Ort::Float16_t> c_BnbDequantize_float16;

    domain.Add(&c_BnbDequantize_float);
    domain.Add(&c_BnbDequantize_float16);
}

}  // namespace Cuda
