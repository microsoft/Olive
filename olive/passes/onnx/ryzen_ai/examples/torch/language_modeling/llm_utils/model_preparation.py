#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Optional, Tuple, Union, Dict
import os
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import psutil

MODEL_NAME_KV_LAYERS_MAP = {
    "mllama": ["*self_attn.k_proj", "*self_attn.v_proj"],
    "llama": ["*k_proj", "*v_proj"],
    "opt": ["*k_proj", "*v_proj"],
    "qwen2moe": ["*k_proj", "*v_proj"],
    "qwen2": ["*k_proj", "*v_proj"],
    "qwen": ["*c_attn"],
    "chatglm": ["*query_key_value"],
    "phi3": ["*qkv_proj"],
    "phi": ["*k_proj", "*v_proj"],
    "mistral": ["*k_proj", "*v_proj"],
    "mixtral": ["*k_proj", "*v_proj"],
    "gptj": ["*k_proj", "*v_proj"],
    "grok": ["*k_proj", "*v_proj"],
    "cohere": ["*k_proj", "*v_proj"],
    "dbrx": ["*Wqkv"],
    "deepseekv2v3": ["*kv_b_proj"],
    "deepseek": ["*k_proj", "*v_proj"],
    "gemma2": ["*k_proj", "*v_proj"]
}

MODEL_NAME_Q_LAYERS_MAP = {
    "mllama": "*self_attn.q_proj",
    "llama": "*q_proj",
    "opt": "*q_proj",
    "qwen2moe": "*q_proj",
    "qwen2": "*q_proj",
    "chatglm": "*query_key_value",
    "phi3": "*qkv_proj",
    "phi": "*q_proj",
    "mistral": "*q_proj",
    "mixtral": "*q_proj",
    "gptj": "*q_proj",
    "grok": "*q_proj",
    "cohere": "*q_proj",
    "dbrx": ["*Wqkv"],
    "deepseek": "*q_proj",
    "deepseekv2v3": ["*q_a_proj", "*q_b_proj"],
}

MODEL_NAME_EXCLUDE_LAYERS_MAP = {
        "mllama": ["*lm_head", "*patch_embedding", "multi_modal_projector"],
        "llama": ["lm_head"],
        "opt": ["lm_head"],
        "qwen2moe": ["lm_head", "*.gate", "*.shared_expert_gate"],
        "qwen2": ["lm_head"],
        "qwen": ["lm_head"],
        "qwq": ["lm_head"],
        "chatglm": ["transformer.output_layer"],
        "phi3": ["lm_head"],
        "phi": ["lm_head"],
        "mistral": ["lm_head"],
        "mixtral": ["lm_head", "*.gate"],
        "gptj": ["lm_head"],
        "grok": ["lm_head", "*.gate"],
        "cohere": ["lm_head"],
        "dbrx": ["lm_head", "*router.layer"],
        "deepseek": ["lm_head", "*.gate"],
        "deepseekv2v3": ["lm_head", "*.gate"],
        "olmo": ["lm_head"],
        "gemma2": ["lm_head"],
        "instella": ["lm_head"],
}

MOE_MODEL_NAME_EXPERTS_LAYERS_MAP = {
    "llama4": ["*feed_forward.experts*", "*feed_forward.shared_expert*"],
    "deepseek": ["*.mlp.experts.*"],
    "grok": ["*.moe_block.experts.*"]
}

MODEL_NAME_PATTERN_MAP = {
        "Mllama": "mllama",
        "Llama": "llama",
        "OPT": "opt",
        "Qwen2Moe": "qwen2moe",
        "QWen2": "qwen2",
        "QWen": "qwen",
        "ChatGLM": "chatglm",
        "Phi3": "phi3",
        "Phi": "phi",
        "Mistral": "mistral",
        "Mixtral": "mixtral",
        "GPTJ": "gptj",
        "Grok": "grok",
        "Cohere": "cohere",
        "dbrx": "dbrx",
        "DeepseekV": "deepseekv2v3",
        "Deepseek": "deepseek",
        "olmo": "olmo",
        "gemma2": "gemma2",
        "instella": "instella",
}

def get_tokenizer(ckpt_path: str, max_seq_len: int = 2048, model_type: Optional[str] = None) -> AutoTokenizer:
    print(f"Initializing tokenizer from {ckpt_path}")
    use_fast = True if model_type in ["grok", "cohere", "olmo", "instella", "deepseekv2v3"] else False
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path,
                                              model_max_length=max_seq_len,
                                              padding_side="left",
                                              trust_remote_code=True,
                                              use_fast=use_fast)
    if model_type and model_type in ["qwen", "qwen2"]:
        # qwen2 use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def prepare_for_moe_quant(model: nn.Module):
    from transformers.models.dbrx.modeling_dbrx import DbrxForCausalLM, DbrxExperts
    from module_replacement.dbrx_expert import DbrxExperts_
    from quark.torch.quantization.utils import set_op_by_name

    if isinstance(model, DbrxForCausalLM):
        for name, module in model.named_modules(remove_duplicate=False):
            if isinstance(module, DbrxExperts):
                new_experts = DbrxExperts_.from_float(module)
                set_op_by_name(model, name, new_experts)
                print(f"module {name} has been replaced")


def get_model(ckpt_path: str, data_type: str = 'auto', device: str = "cuda", multi_gpu: bool = False, multi_device = False, attn_implementation: str = "eager") -> Tuple[nn.Module, torch.dtype]:
    if data_type == 'float16':
        model_dtype = torch.float16
    elif data_type == 'bfloat16':
        model_dtype = torch.bfloat16
    elif data_type == 'float32':
        model_dtype = torch.float32
    elif data_type == 'auto':
        model_dtype = data_type
    else:
        raise ValueError(f"{data_type} not support for current model")
    mllama_list = ["Llama-3.2-11B-Vision", "Llama-3.2-90B-Vision", "Llama-3.2-11B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct"]
    model_name = os.path.basename(os.path.normpath(ckpt_path))
    max_memory = None
    if multi_device:
        device = 'auto'
        max_memory = get_device_max_memory()
    if multi_gpu:
        device = 'auto'
    if model_name in mllama_list:
        from transformers import MllamaForConditionalGeneration
        model = MllamaForConditionalGeneration.from_pretrained(ckpt_path, device_map=device, torch_dtype=model_dtype, max_memory=max_memory, trust_remote_code=True, attn_implementation=attn_implementation)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map=device, torch_dtype=model_dtype, max_memory=max_memory, trust_remote_code=True, attn_implementation=attn_implementation)
        except Exception as e:
            model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map=device, torch_dtype=model_dtype, max_memory=max_memory, trust_remote_code=True)
    if multi_device and hasattr(model, "hf_device_map"):
        print("device_map:", model.hf_device_map)
    # For certain models, the attribute model.config._name_or_path is an empty string; enforce the setting here.
    model.config._name_or_path = ckpt_path

    model.eval()
    model_dtype = next(model.parameters()).dtype

    return model, model_dtype


def get_model_type(model: nn.Module) -> str:
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    print(f"\n[INFO]: This model: {type(model).__name__.lower()} has not been tested with the example provided!")
    print("        There may be risks associated with model loading, algorithm configuration, and exporting.")
    print("        However, this does not mean that Quark definitively does not support this model.")
    print("        If you choose to run this model, please add the model information to the `get_model_type` function in utils/model_preparation.py.")
    exit(2)


def save_model(model: nn.Module, tokenizer: AutoTokenizer, save_dir: str) -> None:
    model.save_pretrained(save_dir, safe_serialization=True)
    if tokenizer is None and getattr(model.config, "_name_or_path", None):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True)
            print(f"Save the tokenizer from pretrained: {model.config._name_or_path}")
        except Exception as e:
            print(f"An error occurred when loading tokenizer: {e}")
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device_max_memory() -> Dict[Union[int, str], Union[int, str]]:
    for i in range(torch.cuda.device_count()):
        _ = torch.tensor([0], device=i)
        cuda_avail_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
        cpu_avail_memory = psutil.virtual_memory().available
        max_memory = {}
        for cuda_num, cuda_memory in cuda_avail_memory.items():
            cuda_memory_gb = cuda_memory / (10**9)
            print(f"GPU{cuda_num} cuda_avail_memory: {cuda_memory_gb:.1f}GB")
            if cuda_num == 0:
                # The ratio is an experience value that you can manually adjust yourself.
                gpu0_ratio = 0.5 if cuda_memory_gb > 30 else 0.3
                max_memory[cuda_num] = f"{cuda_memory_gb * gpu0_ratio:.1f}GB"
            else:
                other_ratio = 0.875 if cuda_memory_gb > 30 else 0.7
                max_memory[cuda_num] = f"{cuda_memory_gb * other_ratio:.1f}GB"
        print(f"cpu_avail_memory: {cpu_avail_memory / (10**9):.1f}GB")
        cpu_ratio = 0.875
        max_memory["cpu"] = f"{cpu_avail_memory / (10**9) * cpu_ratio:.1f}GB"
        print("final_use_model_kwargs: ", max_memory)
        # max_memory =  {0: '0.1GB', 'cpu': '100GB'}

    return max_memory
