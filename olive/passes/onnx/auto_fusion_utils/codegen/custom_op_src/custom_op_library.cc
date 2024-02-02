#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// #include <vector>
// #include <cmath>
#include <mutex>
// #include <system_error>

// #include "core/common/common.h"
// #include "core/framework/ortdevice.h"
// #include "core/framework/ortmemoryinfo.h"
#include "fusion_ops.h"
// #include "onnxruntime_lite_custom_op.h"

static const char* c_OpDomain = "olive.auto_fusion";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
  OrtStatus* result = nullptr;
  try {
    Ort::CustomOpDomain domain{c_OpDomain};
    OliveTritonFusion::RegisterOps(domain);
    Ort::CustomOpDomain domain_v2{"v2"};
    OliveTritonFusion::RegisterOps(domain_v2);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    session_options.Add(domain_v2);
    AddOrtCustomOpDomainToContainer(std::move(domain));
    AddOrtCustomOpDomainToContainer(std::move(domain_v2));
  }
  catch(const std::exception& e) {
    Ort::Status status{e};
    result = status.release();
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
