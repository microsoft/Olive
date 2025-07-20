#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import fnmatch
import torch
from functools import reduce

class SmoothQuantLinearLike(torch.nn.Module):
    def __init__(self, op, scales):
        super(SmoothQuantLinearLike, self).__init__()
        self.op = op
        if self.op.weight.dim() != scales.dim():
            scales = scales.squeeze(0)
        self.op.weight.mul_(scales)
        self.register_buffer('multiplier', scales)

    def forward(self, *args, **kwargs):
        x = args[0]
        x = x / self.multiplier
        x = self.op(x)
        return x

    def __getattr__(self, name):
        if name in self.__dict__.keys():
            return self.__dict__[name]
        if name in self._modules.keys():
            return self._modules[name]
        if name in self._buffers.keys():
            return self._buffers[name]
        if name in self._parameters.keys():
            return self._parameters[name]
        return getattr(self.op, name)

def is_linear_like(op):
    return isinstance(op, torch.nn.Linear) or isinstance(op, torch.nn.Conv2d)

def compute_smoothquant_max_value(op, tensor):
    conv_amax_axis = [x for x in range(tensor.dim()) if x != (tensor.dim() - 3)]
    linear_amax_axis = [x for x in range(tensor.dim()) if x != (tensor.dim() - 1)]
    if isinstance(op, torch.nn.Conv2d):
        activation_max = torch.amax(tensor.abs().detach(), dim=conv_amax_axis, keepdim=True).clamp(min=1e-5)
    elif isinstance(op, torch.nn.Linear):
        activation_max = torch.amax(tensor.abs().detach(), dim=linear_amax_axis, keepdim=True).clamp(min=1e-5)
    else:
        print("smoothquant only support linear and conv2d")
        exit(1)
    return activation_max

def update_activation_max(obj, tensor):
    activation_max = compute_smoothquant_max_value(obj, tensor)
    if not hasattr(obj, 'activation_max'):
        setattr(obj, 'activation_max', activation_max)
    else:
        obj.activation_max = torch.max(obj.activation_max, activation_max)

def smoothquant_forward_hook(module, input, output):
    update_activation_max(module, input[0])

@torch.no_grad()
def calibrate_smoothquant(model, dataloader):
    forward_hooks = []
    for name, module in model.named_modules():
        setattr(module, 'module_name', name)
        if is_linear_like(module):
            forward_hooks.append(module.register_forward_hook(smoothquant_forward_hook))
    count = 0
    for data in dataloader:
        model(data)
        count = count + 1
        print(f"\rsmooth calib:{count}/{len(dataloader)}", end='', flush=True)

    cache_activation_max = {}
    for name, module in model.named_modules():
        if hasattr(module, 'activation_max'):
            cache_activation_max[name] = module.activation_max

    for hook in forward_hooks:
        hook.remove()
    return cache_activation_max

def apply_smoothquant_to_linear_like(model, op, activation_max, alpha=0.9):
    device, dtype = op.weight.device, op.weight.dtype
    activation_max = activation_max.to(device=device, dtype=dtype)
    weight_max = compute_smoothquant_max_value(op, op.weight)
    scales = (activation_max.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype)
    epsilon = 1.0 / (1 << 31)
    if activation_max.min() <= epsilon:
        zero_mask = activation_max <= epsilon
        scales[zero_mask] = 1

    module_list = op.module_name.split('.')
    linear_like_module_name = module_list[-1]
    linear_like_module_parent_module = reduce(getattr, module_list[:-1], model)
    setattr(linear_like_module_parent_module, linear_like_module_name, SmoothQuantLinearLike(op, scales))

@torch.no_grad()
def apply_smoothquant(model, cache_activation_max=None, exclude_layers={}, alpha=0.9):
    def filter_by_name(test_module_name):
        for name_pattern in exclude_layers:
            if fnmatch.fnmatch(test_module_name, name_pattern):
                return True
        return False
    for name, module in model.named_modules():
        setattr(module, 'module_name', name)
        if is_linear_like(module):
            if filter_by_name(name):
                continue
            if name in cache_activation_max.keys():
                activation_max = cache_activation_max[name]
            else:
                activation_max = module.activation_max
            apply_smoothquant_to_linear_like(model, module, activation_max, alpha)
