/*
-------------------------------------------------------------------------
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
--------------------------------------------------------------------------
*/
using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json.Linq;

class Program
{
    static void UpdateSessOptions(SessionOptions sessOptions, JObject sessionOptions)
    {
        int? interOpNumThreads = null;
        if (sessionOptions.ContainsKey("inter_op_num_threads"))
        {
            interOpNumThreads = sessionOptions.Value<int?>("inter_op_num_threads");
        }
        int? intraOpNumThreads = null;
        if (sessionOptions.ContainsKey("intra_op_num_threads"))
        {
            intraOpNumThreads = sessionOptions.Value<int?>("intra_op_num_threads");
        }
        int executionMode = sessionOptions.Value<int>("execution_mode");
        int graphOptimizationLevel = sessionOptions.Value<int>("graph_optimization_level");
        JObject extraSessionConfig = sessionOptions.Value<JObject>("extra_session_config");

        if (interOpNumThreads.HasValue)
        {
            sessOptions.InterOpNumThreads = interOpNumThreads.Value;
        }
        if (intraOpNumThreads.HasValue)
        {
            sessOptions.IntraOpNumThreads = intraOpNumThreads.Value;
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
                sessOptions.AddSessionConfigEntry(kvp.Key, kvp.Value.ToString());
            }
        }
    }

    static void UpdateExecutionProvider(SessionOptions sessOptions, JArray executionProviderJson)
    {
        // Please note that Onnxruntime for C# has limited support for execution providers.
        // Refer to the Onnxruntime documentation for more information.
        // Please specify your device id to each provider if necessary
        string executionProvider = (string)executionProviderJson[0][0];
        JObject epOptionsJson = (JObject)executionProviderJson[0][1];
        Dictionary<string, string> epOptions = epOptionsJson.ToObject<Dictionary<string, string>>();

        if (executionProvider == "CUDAExecutionProvider")
        {
            OrtCUDAProviderOptions providerOptions = new OrtCUDAProviderOptions();
            providerOptions.UpdateOptions(epOptions);
            sessOptions.AppendExecutionProvider_CUDA(providerOptions);
        }

        if (executionProvider == "TensorrtExecutionProvider")
        {
            OrtTensorRTProviderOptions providerOptions = new OrtTensorRTProviderOptions();
            providerOptions.UpdateOptions(epOptions);
            sessOptions.AppendExecutionProvider_Tensorrt(providerOptions);
        }

        if (executionProvider == "OpenVINOExecutionProvider")
        {
            sessOptions.AppendExecutionProvider_OpenVINO();
        }

        if (executionProvider == "SNPEExecutionProvider")
        {
            sessOptions.AppendExecutionProvider("SNPE", epOptions);
        }

        if (executionProvider == "DmlExecutionProvider")
        {
            sessOptions.AppendExecutionProvider_DML();
        }
    }
    static void Main()
    {
        string path = @"c:\path\to\inference_config.json";
        string jsonString = File.ReadAllText(path);
        JObject j = JObject.Parse(jsonString);

        InferenceSession session;
        string modelPath = @"c:\path\to\model.onnx";
        if (j == null)
        {
            // Create inference session
            session = new InferenceSession(modelPath);
        }
        else
        {
            // Get execution_provider section
            JArray executionProvider = (JArray)j["execution_provider"];

            // Get session_options section
            JObject sessOpts = (JObject)j["session_options"];

            // Create inference configuration
            using SessionOptions options = new SessionOptions();
            UpdateSessOptions(options, sessOpts);
            UpdateExecutionProvider(options, executionProvider);

            // Create inference session
            session = new InferenceSession(modelPath, options);
        }

        // Run inference
        // Replace inputs, outputNames with actual variable names
        using var outputTensors = session.Run(<inputs>, <outputsNames>, new RunOptions());

        // Update output logic here
        var outputTensor = outputTensors.First().AsTensor<float>();

        session.Dispose();
    }
}
