# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path

local_model_name = None

_MODEL_TYPE_TO_REPO_ID = {
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "mistral-7b-chat": "mistralai/Mistral-7B-Instruct-v0.1",
    "falcon-7b-chat": "tiiuae/falcon-7b-instruct",
    "llama-2-7b-chat-uncensored": "georgesung/llama2_7b_chat_uncensored",
    "mistral-7b-openorca": "Open-Orca/Mistral-7B-OpenOrca",
    "codellama-7b-chat": "codellama/CodeLlama-7b-Instruct-hf",
    "llava-7b": "llava-hf/llava-1.5-7b-hf",
    "orca-mini-7b": "pankajmathur/orca_mini_7b",
    "vicuna-7b-v1.5": "lmsys/vicuna-7b-v1.5",
    "deepseek-coder-7b-instruct-v1.5": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "wizard-vicuna-7b-uncensored": "cognitivecomputations/Wizard-Vicuna-7B-Uncensored",
    "dolphin-2.6-mistral-7b": "cognitivecomputations/dolphin-2.6-mistral-7b",
    "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
    "openhermes-2.5-mistral-7b": "teknium/OpenHermes-2.5-Mistral-7B",
    "noushermes-llama-2-7b": "NousResearch/Nous-Hermes-llama-2-7b",
    "openchat-7b-3.5": "openchat/openchat_3.5",
    "neural-chat-7b-v3.1": "Intel/neural-chat-7b-v3-1",
    "tinyllama-1.1b-chat-v0.6": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
    "phi-2": "microsoft/phi-2",
    "phi-3-mini-4k": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-mini-128k": "microsoft/Phi-3-mini-128k-instruct",
    "gemma-7b-it": "google/gemma-7b-it",
    "llama-3-8b-it": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def get_supported_lvlm_models():
    return ["llava-7b"]


def get_supported_llm_models():
    return [
        "llama-2-7b-chat",
        "mistral-7b-chat",
        "falcon-7b-chat",
        "llama-2-7b-chat-uncensored",
        "mistral-7b-openorca",
        "codellama-7b-chat",
        "orca-mini-7b",
        "vicuna-7b-v1.5",
        "deepseek-coder-7b-instruct-v1.5",
        "wizard-vicuna-7b-uncensored",
        "dolphin-2.6-mistral-7b",
        "zephyr-7b-beta",
        "openhermes-2.5-mistral-7b",
        "noushermes-llama-2-7b",
        "openchat-7b-3.5",
        "neural-chat-7b-v3.1",
        "tinyllama-1.1b-chat-v0.6",
        "phi-2",
        "phi-3-mini-4k",
        "phi-3-mini-128k",
        "gemma-7b-it",
        "llama-3-8b-it",
    ]


def get_all_supported_models():
    return ["local", *get_supported_lvlm_models(), *get_supported_llm_models()]


def get_model_repo_id(model_type: str):
    global local_model_name  # pylint: disable=global-statement

    if model_type == "local":
        if local_model_name is None:
            local_model_name = input("Enter the full path to your local huggingface model: ")
        return local_model_name

    return _MODEL_TYPE_TO_REPO_ID[model_type]


def get_model_name(model_type: str):
    return get_model_repo_id(model_type).replace("/", "_").replace("\\", "_").replace(":", "_")


def get_model_type(model_name: str):
    for model_type, repo_id in _MODEL_TYPE_TO_REPO_ID.items():
        if repo_id.replace("/", "_") == model_name:
            return model_type

    raise ValueError(f"model_name '{model_name}' not found")


def get_model_dir(model_type: str):
    model_name = get_model_name(model_type)
    script_dir = Path(__file__).resolve().parent
    return script_dir / "models" / "optimized" / model_name
