// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace Cuda {

// template for float and float16
template <typename T>
struct BnbDequantizeKernel {
    BnbDequantizeKernel(const OrtKernelInfo* kernel_info) {
        Ort::ConstKernelInfo info{kernel_info};
        dtype_ = info.GetAttribute<int64_t>("dtype");
        blocksize_ = info.GetAttribute<int64_t>("blocksize");
        quant_type_ = info.GetAttribute<int64_t>("quant_type");
        double_quant_ = info.GetAttribute<int64_t>("double_quantized");
        if (double_quant_) nested_blocksize_ = info.GetAttribute<int64_t>("nested_blocksize");
    }

    void Compute(OrtKernelContext* context);

    private:
        int64_t dtype_;
        int64_t blocksize_;
        int64_t quant_type_;
        int64_t double_quant_;
        int64_t nested_blocksize_;
};

template <typename T>
struct BnbDequantize : Ort::CustomOpBase<BnbDequantize<T>, BnbDequantizeKernel<T>> {
    void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const {
        return new BnbDequantizeKernel<T>(info);
    };

    const char* GetName() const { return "BnbDequantize"; };

    const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };

    size_t GetInputTypeCount() const { return 7; };
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        // First 4 inputs are required, last 3 depends on whether it is double quantized
        if (index >= 4)
            return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

         return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 0)
            return Ort::TypeToTensorType<T>::type;
        else if (index == 1)
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        else if (index == 2)
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        else if (index == 3)
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        else
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return Ort::TypeToTensorType<T>::type; };
};

void RegisterOps(Ort::CustomOpDomain& domain);

}  // namespace Cuda
