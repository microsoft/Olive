#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <mutex>

#include "cuda/matmul_bnb4.h"

static const char* c_OpDomain = "olive";

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
        Cuda::RegisterOps(domain);

        Ort::CustomOpDomain domain_v2{"v2"};
        Cuda::RegisterOps(domain_v2);

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
