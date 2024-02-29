#pragma once

#define ORT_CUDA_CTX

#include "onnxruntime_cxx_api.h"

namespace Ort {
namespace Custom {

enum CudaResource {
  cuda_handle_t = 10000,
};

struct CudaContext {
  static const int cuda_resource_ver = 1;
  void Init(const OrtKernelContext& ctx) {
    const auto& ort_api = Ort::GetApi();
    ort_api.KernelContext_GetResource(&ctx, cuda_resource_ver, CudaResource::cuda_handle_t, &cuda_stream);
    if (!cuda_stream) {
      ORT_CXX_API_THROW("Failed to fetch cuda stream from context", OrtErrorCode::ORT_RUNTIME_EXCEPTION);
    }
  }
  void* cuda_stream = {};
};

}  // namespace Custom
}  // namespace Ort
