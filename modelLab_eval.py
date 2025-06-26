import os
import onnxruntime_genai as og
import time
import json

# https://github.com/microsoft/onnxruntime-genai/blob/main/benchmark/python/benchmark_e2e.py#L424

model_config = "D:\\Downloads\\test-llm\\huggingface_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_v1\\history\\new qpu_T(20250429_134754)\\model_config.json"
text_template = "<｜User｜>{Content}<｜Assistant｜><think>"
prompt_length = 16
generation_length = 64
repetitions = 10
warmup = 5

# prepare dataset

# Use prompt length to get pre-defined prompt
def get_prompt_by_length(prompt_length):
    json_path = "prompts.json"
    with open(json_path, "r") as file:
        data = json.load(file)
    return data[f"{prompt_length}"]

def get_input_tokens(tokenizer, prompt_length, text_template):
    prompt = get_prompt_by_length(prompt_length)
    prompt = text_template.replace("{Content}", prompt)
    return tokenizer.encode(prompt)

# run

def run_one(model, params, input_tokens, tokenizer = None):
    latencies = []
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    # sampling_times?
    # t = time.perf_counter()
    # generator.generate_next_token()
    # latencies.append(time.perf_counter() - t)

    while not generator.is_done():
        t = time.perf_counter()
        generator.generate_next_token()
        latencies.append(time.perf_counter() - t)
    
    if tokenizer:
        print(tokenizer.decode(generator.get_sequence(0)))
    del generator
    return latencies

def run():
    # strange model.model_path is to XX\\cache\\default_workflow\\runs\\10cdfb01\\models
    model_folder = os.path.join(os.path.dirname(model_config), "model")
    model = og.Model(model_folder)
    tokenizer = og.Tokenizer(model)
    search_options = {}
    search_options["max_length"] = prompt_length + generation_length

    input_tokens = get_input_tokens(tokenizer, prompt_length, text_template)
    
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)

    for i in range(warmup):
        run_one(model, params, input_tokens)

    result = []
    for i in range(repetitions):
        latencies = run_one(model, params, input_tokens, tokenizer)
        result.append(latencies)
    return result

latencies = run()

# output result

# D:\Olive\olive\evaluator\olive_evaluator.py
def latency_avg(latencies) -> dict:
    return round(sum(latencies) / len(latencies) * 1000, 5)

metrics_res = {}
flatten_latencies = [item for sublist in latencies for item in sublist]
metrics_res["latency"] = latency_avg(flatten_latencies)
metrics_res["throughput"] = round(1 / metrics_res["latency"] * 1000, 5)
first_latencies = [latency[0] for latency in latencies]
metrics_res["FTL"] = latency_avg(first_latencies)

print(f"Latency: {metrics_res['latency']} ms")
print(f"Throughput: {metrics_res['throughput']} tokens/s")
print(f"FTL: {metrics_res['FTL']} ms")