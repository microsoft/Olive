/*
-------------------------------------------------------------------------
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
--------------------------------------------------------------------------
*/
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include "onnxruntime_cxx_api.h"


void updateSessOptions(SessionOptions& sess_options, const nlohmann::json& session_options);
void updateExecutionProvider(SessionOptions& sess_options, const nlohmann::json& executionProvider_json);


int main() {
    std::ifstream i("inference_config.json");
    const nlohmann::json j = nlohmann::json::parse(i);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    std::unique_ptr<Ort::Session> session;

    if (j.is_null()) {
        session = std::make_unique<Ort::Session>(env, L"model.onnx");
    } else {
        // Get executionProvider section
        const nlohmann::json& executionProvider = j["executionProvider"];

        // Get session_options section
        const nlohmann::json& sessOpts = j["session_options"];

        // Create inference configuration
        Ort::SessionOptions options;
        updateSessOptions(options, sessOpts);
        updateExecutionProvider(options, executionProvider);

        // Create inference session
        session = std::make_unique<Ort::Session>(env, L"model.onnx", options);
    }


    // Run inference
    // Replace inputNames, inputValues, outputNames with actual variable names
    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames, inputValues, 1, outputNames, 1);

    // Get output tensor
    auto outputTensor = outputTensors.front().Get<Tensor>();
}


void updateSessOptions(SessionOptions& sessOptions, const nlohmann::json& sessionOptionsJson) {
    int interOpNumThreads = sessionOptionsJson.value("inter_op_num_threads", -1);
    int intraOpNumThreads = sessionOptionsJson.value("intra_op_num_threads", -1);
    int executionMode = sessionOptionsJson.value("execution_mode", -1);
    int graphOptimizationLevel = sessionOptionsJson.value("graph_optimization_level", -1);
    nlohmann::json extraSessionConfig = sessionOptionsJson.value("extra_session_config", nlohmann::json::object());

    if (intraOpNumThreads != -1) {
        sessOptions.SetIntraOpNumThreads(intraOpNumThreads);
    }
    if (interOpNumThreads != -1) {
        sessOptions.SetInterOpNumThreads(interOpNumThreads);
    }
    if (executionMode != -1) {
        if (executionMode == 0) {
            sessOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        } else if (executionMode == 1) {
            sessOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        }
    }
    if (graphOptimizationLevel != -1) {
        sessOptions.SetGraphOptimizationLevel(GraphOptimizationLevel(graphOptimizationLevel));
    }
    if (!extraSessionConfig.empty()) {
        for (auto& [key, value] : extraSessionConfig.items()) {
            sessOptions.AddConfigEntry(key, value.dump());
        }
    }
}

void updateExecutionProvider(SessionOptions& sessOptions, const nlohmann::json& executionProviderJson) {

    std::string executionProvider = executionProviderJson.at(0).at(0)
    nlohmann::json epOptions = executionProviderJson.at(0).at(1)

    // Please check https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html for more details
    if (executionProvider == "CUDAExecutionProvider") {
        OrtCUDAProviderOptionsV2 providerOptions;

        // Update your configurations here
        // providerOptions.device_id = epOptions.value("device_id");

        sessOptions.AppendExecutionProvider_CUDA_V2(providerOptions);
    }

    // Please check https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html for more details
    if (executionProvider == "TensorrtExecutionProvider") {
        OrtTensorRTProviderOptions providerOptions;

        // Update your configurations here
        // providerOptions.device_id = epOptions.value("device_id");;

        sessOptions.AppendExecutionProvider_TensorRT(providerOptions);
    }

    // Please check https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html for more details
    if (executionProvider == "OpenVINOExecutionProvider") {
        OrtOpenVINOProviderOptions providerOptions;

        // Update your configurations here
        // providerOptions.device_type = epOptions.value("device_type");;

        sessOptions.AppendExecutionProvider_OpenVINO(providerOptions);
    }

    // Please check https://onnxruntime.ai/docs/execution-providers/SNPE-ExecutionProvider.html for more details
    if (executionProvider == "SNPEExecutionProvider") {
        std::unordered_map<std::string, std::string> providerOptions;

        // Update your configurations here
        // providerOptions["runtime"] = "DSP";
        // providerOptions["buffer_type"] = "FLOAT";

        sessOptions.AppendExecutionProvider("SNPE", providerOptions);
    }
}
