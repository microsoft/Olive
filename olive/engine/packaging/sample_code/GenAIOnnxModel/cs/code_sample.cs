/*
-------------------------------------------------------------------------
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
--------------------------------------------------------------------------
*/
using Microsoft.ML.OnnxRuntimeGenAI;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Diagnostics;

string modelPath = args[0];

Console.WriteLine("Loading model ...");
using Model model = new(modelPath);

Console.WriteLine("Creating tokenizer ...");
using Tokenizer tokenizer = new(model);

Console.WriteLine("Loading genai_config.json ...");
string config_filepath = Path.Combine(modelPath, "genai_config.json");
JObject? config;
using (StreamReader file = File.OpenText(config_filepath))
{
    using JsonTextReader reader = new(file);
    config = (JObject)JToken.ReadFrom(reader);
}

Console.WriteLine("Evaluating generator params and search options ...");
using GeneratorParams generatorParams = new(model);
JToken? search;
if (config.TryGetValue("search", out search))
{
    foreach (var entry in (JObject)search)
    {
        switch (entry.Key)
        {
        case "diversity_penalty": generatorParams.SetSearchOption("diversity_penalty", double.Parse((string)entry.Value)); break;
        case "do_sample": generatorParams.SetSearchOption("do_sample", bool.Parse((string)entry.Value)); break;
        case "early_stopping": generatorParams.SetSearchOption("early_stopping", bool.Parse((string)entry.Value)); break;
        case "length_penalty": generatorParams.SetSearchOption("length_penalty", double.Parse((string)entry.Value)); break;
        case "max_length": generatorParams.SetSearchOption("max_length", double.Parse((string)entry.Value)); break;
        case "min_length": generatorParams.SetSearchOption("min_length", double.Parse((string)entry.Value)); break;
        case "no_repeat_ngram_size": generatorParams.SetSearchOption("no_repeat_ngram_size", double.Parse((string)entry.Value)); break;
        case "num_beams": generatorParams.SetSearchOption("num_beams", double.Parse((string)entry.Value)); break;
        case "num_return_sequences": generatorParams.SetSearchOption("num_return_sequences", double.Parse((string)entry.Value)); break;
        case "past_present_share_buffer": generatorParams.SetSearchOption("past_present_share_buffer", bool.Parse((string)entry.Value)); break;
        case "repetition_penalty": generatorParams.SetSearchOption("repetition_penalty", double.Parse((string)entry.Value)); break;
        case "temperature": generatorParams.SetSearchOption("temperature", double.Parse((string)entry.Value)); break;
        case "top_k": generatorParams.SetSearchOption("top_k", double.Parse((string)entry.Value)); break;
        case "top_p": generatorParams.SetSearchOption("top_p", double.Parse((string)entry.Value)); break;
        }
    }
}

Console.WriteLine("Encoding prompts ...");
const string prompt = "Who is Albert Einstein?";
using var sequences = tokenizer.Encode(prompt);
generatorParams.SetInputSequences(sequences);

Console.WriteLine("Generating tokens ...");
var watch = Stopwatch.StartNew();
var outputSequences = model.Generate(generatorParams);
var runTime = watch.Elapsed.TotalSeconds;

Console.WriteLine("Decoding generated tokens ...");
var answer = tokenizer.Decode(outputSequences[0]);

Console.WriteLine("Prompt: " + prompt);
Console.WriteLine("Output: " + answer);
Console.WriteLine(
    string.Format("Tokens: {0}, Time: {1:.02} seconds, Tokens/second: {2:.02}",
    outputSequences[0].Length, runTime, outputSequences[0].Length / (double)runTime));
