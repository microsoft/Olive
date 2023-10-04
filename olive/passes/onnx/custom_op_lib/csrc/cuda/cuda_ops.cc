// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <iostream>
#include <cuda_runtime.h>

#include "cuda_ops.h"


namespace Cuda {

void BnbDequantizeKernel::Compute(OrtKernelContext* context) {
    // first input is not used currently
    // keeping it to infer the type of the weight
    // might also help execution order so that the BnbDequantize node has a parent
    // TODO(jambayk): clean this up so that we don't need to pass the first input
    Ort::KernelContext ctx(context);
    auto B_quant = ctx.GetInput(0);
    auto B_shape = ctx.GetInput(2);

    const float_t* absmax_value;
    if (double_quant_) {
        auto absmax_int8 = ctx.GetInput(1);
        auto offset = ctx.GetInput(3);
        auto nested_absmax = ctx.GetInput(4);
        auto nested_code = ctx.GetInput(5);

        // TODO(jambayk): dequantize absmax_int8, move the value to device, return pointer
    } else {
        auto absmax_float = ctx.GetInput(1);
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
    // typedef jk if (dtype_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) { float_t; } else { Ort::; })
    void* B_dequant_data;
    switch (dtype_)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        B_dequant_data = B_dequant.GetTensorMutableData<Ort::Float16_t>();
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        B_dequant_data = B_dequant.GetTensorMutableData<Ort::BFloat16_t>();
        break;
    default:
        B_dequant_data = B_dequant.GetTensorMutableData<float_t>();
        break;
    }
}

void RegisterOps(Ort::CustomOpDomain& domain) {
    static const BnbDequantize c_BnbDequantize;

    domain.Add(&c_BnbDequantize);
}

}  // namespace Cuda
