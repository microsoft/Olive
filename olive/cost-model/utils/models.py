# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from layers import Embedding, Linear, Dropout, SiLU
from layers import LlamaRotaryEmbedding, LlamaRMSNorm
from layers import Phi3RMSNorm

class Model:
    def __init__(self):
        self._layers = []
        self._kvc = 0
        self._bs = 1
        self._weight_precision = "fp16"
        self._kv_precision = "fp16"

    def add_layer(self, l, count = 1):
        self._layers.append((l,count))

    def param_count(self):
        pc = 0
        for (l,c) in self._layers:
            pc = pc + l.param_count()*c
        return pc

    def flops(self, nt = 1):
        fc = 0
        for (l,c) in self._layers:
            fc = fc + l.flops()*c
        return fc * nt

    def set_batch_size(self, bs):
        self._bs = bs
        for (l,_) in self._layers:
            l.set_batch_size(bs)
        return

    # kvc = 2 * nlayers * nheads * dhead
    def set_kv_element_count(self, kvc):
        self._kvc = kvc

    def kv_size(self, n_token = 1):
        return self._kvc * self._bs * n_token * self.get_bytes_for_precision(self._kv_precision)

    def get_bytes_for_precision(self, precision):
        p = precision.upper()
        if p == "FP32": return 4
        if p == "FP16": return 2
        if p == "FP8": return 1
        if p == "FP4" or "NF4": return 0.5
        if p == "INT8" or "UINT8": return 1
        if p == "INT16" or "UINT16": return 2
        if p == "INT32" or "UINT32": return 4

    def memory_for_token_processing(self, ntokens):
        return (self.get_bytes_for_precision(self._weight_precision) * self.param_count()) \
                + self.kv_size(ntokens)
    
    def print_cost_model(self, input_len, output_len):
        wp = self.get_bytes_for_precision(self._weight_precision)
        print("layer,params,flops,weights_memory")
        print("-----,------,-----,--------------")
        for (l,_) in self._layers:
            print(f"{l.__class__.__name__},{l.param_count()},{l.flops()},{l.param_count()*wp}")
        print("\nprompt_memory,token_gen_memory")
        print("-------------,----------------")
        print(f"{self.memory_for_token_processing(input_len)},{self.memory_for_token_processing(input_len+output_len)}")


def get_llama2_7b():
    llama2 = Model()

    # Embedding
    llama2.add_layer(Embedding(32000, 4096))

    # LlamaAttention
    # Q_proj
    llama2.add_layer(Linear(4096,4096), 32)
    # K_proj
    llama2.add_layer(Linear(4096,4096), 32)
    # V_proj
    llama2.add_layer(Linear(4096,4096), 32)
    # O_proj
    llama2.add_layer(Linear(4096,4096), 32)
    # Rotaery Embedding
    llama2.add_layer(LlamaRotaryEmbedding(), 32)

    llama2.set_kv_element_count(2*32*4096)


    # LlamaMLP
    # gate_proj
    llama2.add_layer(Linear(4096,11008), 32)
    # up_proj
    llama2.add_layer(Linear(4096,11008), 32)
    # down_proj
    llama2.add_layer(Linear(11008,4096), 32)

    # input_layernorm
    llama2.add_layer(LlamaRMSNorm(4096), 32)
    # post_attention_layernorm
    llama2.add_layer(LlamaRMSNorm(4096), 32)

    # norm
    llama2.add_layer(LlamaRMSNorm(4096))

    # lm_head
    llama2.add_layer(Linear(4096,32000))

    return llama2

def get_llama2_13b():
    llama2 = Model()

    # Embedding
    llama2.add_layer(Embedding(32000, 5120))

    # LlamaAttention
    # Q_proj
    llama2.add_layer(Linear(5120,5120), 40)
    # K_proj
    llama2.add_layer(Linear(5120,5120), 40)
    # V_proj
    llama2.add_layer(Linear(5120,5120), 40)
    # O_proj
    llama2.add_layer(Linear(5120,5120), 40)
    # Rotaery Embedding
    llama2.add_layer(LlamaRotaryEmbedding(), 40)

    llama2.set_kv_element_count(2*40*5120)

    # LlamaMLP
    # gate_proj
    llama2.add_layer(Linear(5120,13824), 40)
    # up_proj
    llama2.add_layer(Linear(5120,13824), 40)
    # down_proj
    llama2.add_layer(Linear(13824,5120), 40)

    # input_layernorm
    llama2.add_layer(LlamaRMSNorm(5120), 40)
    # post_attention_layernorm
    llama2.add_layer(LlamaRMSNorm(5120), 40)

    # norm
    llama2.add_layer(LlamaRMSNorm(5120))

    # lm_head
    llama2.add_layer(Linear(5120,32000))

    return llama2

def get_phi3_mini_4k():
    phi3 = Model()

    # Embedding
    phi3.add_layer(Embedding(32064, 3072, padding_idx=32000))

    # Dropout
    phi3.add_layer(Dropout())

    # Phi3Attention
    # o_proj
    phi3.add_layer(Linear(3072,3072), 32)
    # qkv_proj
    phi3.add_layer(Linear(3072,9216), 32)
    # Rotaery Embedding
    phi3.add_layer(LlamaRotaryEmbedding(), 32)

    phi3.set_kv_element_count(2*32*3072)

    # Phi3MLP
    # gate_up_proj
    phi3.add_layer(Linear(3072, 16386), 32)
    # down_proj
    phi3.add_layer(Linear(8192, 3072), 32)
    # activation
    phi3.add_layer(SiLU(3072), 32)
    # input_layernorm
    phi3.add_layer(Phi3RMSNorm(3072), 32)
    # resid_attn_dropout
    phi3.add_layer(Dropout(), 32)
    # resid_mlp__dropout
    phi3.add_layer(Dropout(), 32)
    # post_attention_layer_norm
    phi3.add_layer(Phi3RMSNorm(3072), 32)

    # norm
    phi3.add_layer(Linear(3072, 32064))
    return phi3
