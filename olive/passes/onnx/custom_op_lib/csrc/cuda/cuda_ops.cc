// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <numeric>

#include "common.h"
#include "cuda_ops.h"
#include "cuda_ops.cuh"


namespace Cuda {

template <typename T>
void BnbDequantizeKernel<T>::Compute(OrtKernelContext* context) {
    // first input is not used currently
    // keeping it to infer the type of the weight
    // might also help execution order so that the BnbDequantize node has a parent
    // TODO(jambayk): clean this up so that we don't need to pass the first input
    Ort::KernelContext ctx(context);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());

    auto B_quant = ctx.GetInput(1);
    auto B_shape = ctx.GetInput(3);

    const float_t* absmax_value;
    float_t* absmax_value_empty;
    if (double_quant_) {
        auto absmax_int8 = ctx.GetInput(2);
        auto offset = ctx.GetInput(4);
        auto nested_absmax = ctx.GetInput(5);
        auto nested_code = ctx.GetInput(6);

        // TODO(jambayk): see if ort api has easier way to get
        int absmax_int8_numel = absmax_int8.GetTensorTypeAndShapeInfo().GetElementCount();
        cudaMalloc(&absmax_value_empty, sizeof(float_t) * absmax_int8_numel);

        // dequantize nested absmax
        dequantizeBlockwise<float_t, General8bit>(nested_code.GetTensorData<float_t>(), absmax_int8.GetTensorData<uint8_t>(),
                                                  nested_absmax.GetTensorData<float_t>(), absmax_value_empty, nested_blocksize_,
                                                  absmax_int8_numel, stream);
        // add offset to nested absmax
        addOffset(absmax_value_empty, offset.GetTensorData<float_t>(), absmax_int8_numel, stream);
        absmax_value = absmax_value_empty;
    } else {
        auto absmax_float = ctx.GetInput(2);
        absmax_value = absmax_float.GetTensorData<float_t>();
    }

    // get shape of output
    // only support 2D weights so B_shape_size is always 2, just to be safe
    size_t B_shape_size = B_shape.GetTensorTypeAndShapeInfo().GetElementCount();
    int64_t* B_shape_local = new int64_t[B_shape_size];
    cudaMemcpyAsync(B_shape_local, B_shape.GetTensorData<int64_t>(), B_shape_size * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);

    // get output and allocate memory
    // TODO(jambayk): currently, output type is same as T. It is forced in the quantizer pass
    // find ways to cast to T if this is not the case in the future
    auto B_dequant = ctx.GetOutput(0, B_shape_local, B_shape_size);
    void* B_dequant_data = B_dequant.GetTensorMutableData<T>();
    // could get using B_dequant.GetTensorTypeAndShapeInfo().GetElementCount()
    // but want to calulate using B_shape_local to be safe
    int B_dequant_numel = std::accumulate(B_shape_local, B_shape_local + B_shape_size, 1LL, std::multiplies<int64_t>());
    delete[] B_shape_local;

    // dequantize using cuda kernel
    const uint8_t* B_quant_data = B_quant.GetTensorData<uint8_t>();
    if (std::is_same_v<T, Ort::Float16_t>) {
        half* out = static_cast<half*>(B_dequant_data);
        switch (quant_type_)
        {
        case 1:
            dequantizeBlockwise<half, FP4>(nullptr, B_quant_data, absmax_value, out, blocksize_, B_dequant_numel, stream);
            break;
        case 2:
            dequantizeBlockwise<half, NF4>(nullptr, B_quant_data, absmax_value, out, blocksize_, B_dequant_numel, stream);
            break;
        default:
            std::cout << "Unsupported quant_type for half" << quant_type_ << std::endl;
            // this should never happen
            break;
        }
        return;
    }
    else {
        T* out = static_cast<T*>(B_dequant_data);
        switch (quant_type_)
        {
        case 1:
            dequantizeBlockwise<T, FP4>(nullptr, B_quant_data, absmax_value, out, blocksize_, B_dequant_numel, stream);
            break;
        case 2:
            dequantizeBlockwise<T, NF4>(nullptr, B_quant_data, absmax_value, out, blocksize_, B_dequant_numel, stream);
            break;
        default:
            // this should never happen
            break;
        }
    }

    // free memory
    if (double_quant_) {
        cudaFree(absmax_value_empty);
    }
}

void RegisterOps(Ort::CustomOpDomain& domain) {
    static const BnbDequantize<float_t> c_BnbDequantize_float;
    static const BnbDequantize<Ort::Float16_t> c_BnbDequantize_float16;

    domain.Add(&c_BnbDequantize_float);
    domain.Add(&c_BnbDequantize_float16);
}

}  // namespace Cuda
