// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <iostream>
#include <numeric>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


#include "common.h"
#include "matmul_bnb4.h"
#include "matmul_bnb4.cuh"
#include "dequantize_blockwise_bnb4.cuh"

template <typename T>
class ToCudaType {
 public:
  typedef T MappedType;
};

template <>
class ToCudaType<Ort::Float16_t> {
 public:
  typedef half MappedType;
};

namespace Cuda {

template <typename T>
void MatMulBnb4Kernel<T>::Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());

    auto A = ctx.GetInput(0);
    auto B_quant = ctx.GetInput(1);
    auto data_type = ctx.GetInput(3);

    const auto* A_data = A.GetTensorData<T>();
    const uint8_t* B_quant_data = B_quant.GetTensorData<uint8_t>();
    const float* data_type_data = data_type.GetTensorData<float>();

    const float_t* absmax_data;
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
        absmax_data = absmax_value_empty;
    } else {
        auto absmax_float = ctx.GetInput(2);
        absmax_data = absmax_float.GetTensorData<float_t>();
    }

    // get shape of A: M1 X ... X Mn X K.
    // Ex. batch_size X seq_len X K
    const std::vector<int64_t> A_shape = A.GetTensorTypeAndShapeInfo().GetShape();
    // stack all dimensions except the last one
    int64_t M = std::accumulate(A_shape.begin(), A_shape.end() - 1, 1, std::multiplies<int64_t>());
    // // ensure that the last dimension of A is equal to K
    // int64_t K = A_shape.back();
    // assert(K == K_);

    typedef typename ToCudaType<T>::MappedType CudaT;

    // shape of output: M1 X ... X Mn X N
    int64_t out_shape[A_shape.size()];
    std::copy(A_shape.begin(), A_shape.end(), out_shape);
    out_shape[A_shape.size() - 1] = N_;
    auto output = ctx.GetOutput(0, static_cast<const int64_t*>(out_shape), A_shape.size());

    bool is_4bit_done = TryMatMulBnb4(
        reinterpret_cast<CudaT*>(output.GetTensorMutableData<T>()),
        reinterpret_cast<const CudaT*>(A_data),
        B_quant_data,
        absmax_data,
        data_type_data,
        M,
        N_,
        K_,
        blocksize_,
        stream);
    if (is_4bit_done) {
        std::cout << "4bit done" << std::endl;
    }
    else {
        std::cout << "4bit not done" << std::endl;
    }

    // void* output_data = output.GetTensorMutableData<T>();

    // bool is_4bit_done;
    // if (std::is_same_v<T, Ort::Float16_t>) {
    //     is_4bit_done = TryMatMulBnb4<half, 16>(
    //         output.GetTensorMutableData<T>(),
    //         A.GetTensorData<T>(),
    //         B_quant.GetTensorData<uint8_t>(),
    //         absmax_value,
    //         nullptr,
    //         M,
    //         N_,
    //         K_,
    //         blocksize_,
    //         stream);
    // } else {
    //     is_4bit_done = TryMatMulBnb4<T, 32>(
    //         output.GetTensorMutableData<T>(),
    //         A.GetTensorData<T>(),
    //         B_quant.GetTensorData<uint8_t>(),
    //         absmax_value,
    //         nullptr,
    //         M,
    //         N_,
    //         K_,
    //         blocksize_,
    //         stream);
    // }

    // // get output and allocate memory
    // // TODO(jambayk): currently, output type is same as T. It is forced in the quantizer pass
    // // find ways to cast to T if this is not the case in the future
    // int64_t B_dequant_shape[2] = {N_, K_};
    // auto B_dequant = ctx.GetOutput(0, B_dequant_shape, 2);
    // void* B_dequant_data = B_dequant.GetTensorMutableData<T>();

    // // dequantize using cuda kernel
    // const uint8_t* B_quant_data = B_quant.GetTensorData<uint8_t>();
    // if (std::is_same_v<T, Ort::Float16_t>) {
    //     half* out = static_cast<half*>(B_dequant_data);
    //     switch (quant_type_)
    //     {
    //     case 1:
    //         dequantizeBlockwise<half, FP4>(nullptr, B_quant_data, absmax_value, out, blocksize_, N_ * K_, stream);
    //         break;
    //     case 2:
    //         dequantizeBlockwise<half, NF4>(nullptr, B_quant_data, absmax_value, out, blocksize_, N_ * K_, stream);
    //         break;
    //     default:
    //         std::cout << "Unsupported quant_type for half" << quant_type_ << std::endl;
    //         // this should never happen
    //         break;
    //     }
    //     return;
    // }
    // else {
    //     T* out = static_cast<T*>(B_dequant_data);
    //     switch (quant_type_)
    //     {
    //     case 1:
    //         dequantizeBlockwise<T, FP4>(nullptr, B_quant_data, absmax_value, out, blocksize_, N_ * K_, stream);
    //         break;
    //     case 2:
    //         dequantizeBlockwise<T, NF4>(nullptr, B_quant_data, absmax_value, out, blocksize_, N_ * K_, stream);
    //         break;
    //     default:
    //         // this should never happen
    //         break;
    //     }
    // }

    // free memory
    if (double_quant_) {
        cudaFree(absmax_value_empty);
    }
}

void RegisterOps(Ort::CustomOpDomain& domain) {
    static const MatMulBnb4<float_t> c_MatMulBnb4_float;
    static const MatMulBnb4<Ort::Float16_t> c_MatMulBnb4_float16;

    domain.Add(&c_MatMulBnb4_float);
    domain.Add(&c_MatMulBnb4_float16);
}

}  // namespace Cuda
