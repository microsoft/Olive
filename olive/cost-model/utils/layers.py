# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
class Layer:
    def __init__(self):
        self._batch_size = 1
        return

    def set_batch_size(self,b):
        self._batch_size = b
        return
    
    def param_count(self):
        return 0

    def memory(self, percision):
        return 0
    
    def flops(self):
        return 0
    
class Embedding(Layer):
    def __init__(self, v, h, padding_idx=0):
        self._vocab_size = v
        self._hidden_size = h
        self._padding_idx = padding_idx
        return

    def param_count(self):
        pc = self._vocab_size * self._hidden_size
        return pc
        
class Linear(Layer):
    def __init__(self, i, o):
        super().__init__()
        self._in_features = i
        self._out_features = o
        return
    
    def param_count(self):
        pc = self._in_features * self._out_features
        return pc
    
    def flops(self):
        return 2 * self._in_features * self._out_features * self._batch_size

class LlamaRotaryEmbedding(Layer):
    def __init__(self):
        super().__init__()
        return
    
class LayerNorm(Layer):
    def __init__(self, i):
        super().__init__()
        self._i = i
        return
    
    def param_count(self):
        pc = self._i
        return pc
    
    def flops(self):
        return 4 * self._i * self._batch_size

class LlamaRMSNorm(LayerNorm):
    def __init__(self, i):
        super().__init__(i)
        return
    
class Phi3RMSNorm(LayerNorm):
    def __init__(self, i):
        super().__init__(i)
        return
    
class Dropout(Layer):
    def __init__(self):
        super().__init__()
        return
    
class SiLU(Layer):
    def __init__(self, i):
        super().__init__()
        self._i = i
        return
    
    def param_count(self):
        pc = self._i
        return pc
    
    def flops(self):
        return self._i * self._batch_size

