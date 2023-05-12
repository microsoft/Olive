/*
-------------------------------------------------------------------------
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
--------------------------------------------------------------------------
*/
using System;
using System.IO;
using Newtonsoft.Json.Linq;
using Onnxruntime;

class Program
{
    static void UpdateSessOptions(InferenceSessionConfiguration sessOptions, JObject sessionOptions)
    {
        int interOpNumThreads = sessionOptions.Value<int>("inter_op_num_threads");
        int intraOpNumThreads = sessionOptions.Value<int>("intra_op_num_threads");
        int executionMode = sessionOptions.Value<int>("execution_mode");
        int graphOptimizationLevel = sessionOptions.Value<int>("graph_optimization_level");
        JObject extraSessionConfig = sessionOptions.Value<JObject>("extra_session_config");

        if (interOpNumThreads != -1)
        {
            sessOptions.InterOpNumThreads = interOpNumThreads;
        }
        if (intraOpNumThreads != -1)
        {
            sessOptions.IntraOpNumThreads = intraOpNumThreads;
        }
        if (executionMode != -1)
        {
            if (executionMode == 0)
            {
                sessOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            }
            else if (executionMode == 1)
            {
                sessOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            }
        }
        if (graphOptimizationLevel != -1)
        {
            sessOptions.GraphOptimizationLevel = (GraphOptimizationLevel)graphOptimizationLevel;
        }
        if (extraSessionConfig != null)
        {
            foreach (var kvp in extraSessionConfig)
            {
                sessOptions[kvp.Key] = kvp.Value.ToString();
            }
        }
    }

    static void UpdateExecutionProvider(InferenceSessionConfiguration sessOptions, JArray executionProviderJson)
    {
        // Please note that Onnxruntime for C# has limited support for execution providers.
        // Refer to the Onnxruntime documentation for more information.
        executionProvider = (string)executionProviderJson[0][0];
        epOptionsJson = (JObject)executionProviderJson[0][1]
        Dictionary<string, string> epOptions = epOptionsJson.ToObject<Dictionary<string, string>>();

        if (executionProvider == "CUDAExecutionProvider")
        {
            sessOptions.AppendExecutionProvider_CUDA(epOptions);
        }

        if (executionProvider == "TensorrtExecutionProvider")
        {
            sessOptions.AppendExecutionProvider_Tensorrt(epOptions);
        }

        if (executionProvider == "OpenVINOExecutionProvider")
        {
            sessOptions.AppendExecutionProvider_OpenVINO(epOptions);
        }

        if (executionProvider == "SNPEExecutionProvider")
        {
            sessOptions.AppendExecutionProvider("SNPE", epOptions);
        }
    }

    static void Main()
    {
        string jsonString = File.ReadAllText("inference_config.json");
        JObject j = JObject.Parse(jsonString);

        InferenceSession session;

        if (j == null)
        {
            // Create inference session
            session = new InferenceSession("model.onnx");
        }
        else
        {
            // Get execution_provider section
            JArray executionProvider = (JArray)j["execution_provider"];

            // Get session_options section
            JObject sessOpts = (JObject)j["session_options"];

            // Create inference configuration
            using InferenceSessionConfiguration options = new InferenceSessionConfiguration();
            UpdateSessOptions(options, sessOpts);
            UpdateExecutionProvider(options, executionProvider);

            // Create inference session
            session = new InferenceSession("model.onnx", options);
        }

        // Run inference
        // Replace inputNames, inputValues, outputNames with actual variable names
        using var outputTensors = session.Run(new RunOptions(), inputNames, inputValues, outputNames);

        // Get output tensor
        var outputTensor = outputTensors.First().AsTensor<float>();

        session.Dispose();
    }
}
