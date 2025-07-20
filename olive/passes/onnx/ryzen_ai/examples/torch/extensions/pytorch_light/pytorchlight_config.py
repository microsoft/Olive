#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union, Optional, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass, field, make_dataclass

from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.quantization.config.config import Config
from quark.torch.quantization.config.type import Dtype
from quark.torch.export.config.config import ExporterConfig

from pytorchlight.config.configs_sets import int_k
from pytorchlight.nn import (LLinear, LLayerNorm)


from pytorchlight.runtime import auto_replace_to_light, prepare
from pytorchlight.optimizer.calib.base import export_calib_value, load_calib_value

from pytorchlight.export import ONNXQDQManager
from pytorchlight.optimizer.brecq.brecq import BRECQ as pytorchlight_BRECQ
from pytorchlight.config.global_config import quant_configs



@dataclass
class BRECQ:
    pass

class PytorchlightMode(Enum):
    INT_K = "int_k"
    BFP16 = "bfp16"

class PytorchlightConfig:
    def __init__(self, mode: PytorchlightMode, init_config: Dict, calibrate_method: str, mapping: Dict):
        if mode == PytorchlightMode.INT_K:
            int_k()

        for k, v in init_config.items():
            quant_configs[k] = v
        self.mapping = mapping
        # 'Default: MaxMin. Opt: MaxMin, AvgMovingMaxMin, Percentile and MSE'
        self.calibrate_method = calibrate_method
        self.brecq = None

    @staticmethod
    def create_from_quark_config(config: Config):
        pytorch_light_config = {}
        dtype_list = []
        if config.global_quant_config.input_tensors is not None:
            dtype_list.append(config.global_quant_config.input_tensors.dtype)

        if config.global_quant_config.output_tensors is not None:
            dtype_list.append(config.global_quant_config.output_tensors.dtype)

        if config.global_quant_config.weight is not None:
            dtype_list.append(config.global_quant_config.weight.dtype)

        if config.global_quant_config.bias is not None:
            dtype_list.append(config.global_quant_config.bias.dtype)


        dtype_bool_list = []
        for x in dtype_list:
            dtype_bool_list.append((dtype_list[0] == x))

        assert(all(dtype_bool_list))
        global_dtype = dtype_list[0]
        if global_dtype == Dtype.bfp16:

            # seting global
            pytorch_light_config = {'quant_mode.global': 'bfpquantizer', 'quant_bit.global': 8, 'quant_bfp_tile_size.global': 8, 'quant_round_mode.global': 'ROUND_EVEN'}
            mapping_for_quark = {
                nn.Linear: LLinear,
                # nn.Softmax: LSoftmax,  # pytorch-light bug, LSoftmax can not export
                nn.LayerNorm: LLayerNorm
            }
            pytorchlight_config_obj = PytorchlightConfig(PytorchlightMode.BFP16, pytorch_light_config, 'MaxMin', mapping_for_quark)

        if global_dtype == Dtype.int8:
            pytorch_light_config = {'quant_calib_samples': 277, 'quant_bit.global': 8, 'quant_round_mode.global': 'ROUND_EVEN', 'quant_sym_mode.global': True,
                                    'quant_narrow_range.global': False, 'quant_scale_pot_round_mode.global': 'ROUND_CEIL', 'quant_use_qoperator.global': False,
                                    'quant_granularity.global': 'per-tensor', 'quant_group_channel_size.global': 1, 'quant_group_channel_axis.global': None, 'quant_signed_mode.global': True,
                                    'quant_scale.global': None, 'quant_zero_point.global': None, 'quant_scale_type.global': 'fp32', 'quant_scale_quant_int_bit.global': 16,
                                    'quant_scale_quant_int_scale_type.global': 'PoT', 'quant_int_tile_size': 128, 'quant_int_tile_axis': 0, 'quant_group_for_group_conv': None,
                                    'quant_dynamic_mode.global': False, 'quant_mode.input': 'InputINTQuantizer', 'quant_mode.weight': 'WeightINTQuantizer',
                                    'quant_mode.input_x': 'InputINTQuantizer', 'quant_mode.input_y': 'InputINTQuantizer', 'quant_granularity.weight': 'per-tensor',
                                    'quant_optimizer_mode.global': {'InputINTQuantizer': ['MaxMin'], 'WeightINTQuantizer': ['MaxMin']},
                                    'quant_calib_granularity': 'all-heads', 'quant_calib_mse_bins': 0.0, 'quant_calib_percentile_percentage': 0.0}

            pytorchlight_config_obj = PytorchlightConfig(PytorchlightMode.INT_K, pytorch_light_config, 'MaxMin', mapping = {nn.Linear: LLinear})

        if isinstance(config.algo_config, BRECQ):
            pytorchlight_config_obj.brecq = {}
            pytorchlight_config_obj.brecq['brecq_params_config'] = {'num_batches': 20, 'batch_size': 1, 'iters': 1000, 'initial_lr': 0.001, 'drop_ratio': 0.95, 'is_block_recon': True, 'dummy_input_path': 'dummy_input_1'}
        return pytorchlight_config_obj

def extend_enum(enum_obj, name):
    dtype_dict = {x : x for x in enum_obj.__members__}
    if name not in enum_obj.__members__.keys():
        dtype_dict[name] = name
    return Enum(Dtype.__name__, dtype_dict)

def extend_dataclass(class_obj, name, data_type, default_data):
    tmp_new = []
    for k, v in class_obj.__dataclass_fields__.items():
        # there is not reasonable
        tmp_new.append((k, v.type, field(default=None, init=True)))
    tmp_new.append((name, data_type, field(default=default_data, init=True)))
    return make_dataclass(class_obj.__name__, tmp_new)

Dtype = extend_enum(Dtype, 'bfp16')
ExporterConfig = extend_dataclass(ExporterConfig, 'pytorch_light_export_config', ExporterConfig, None)


class PytorchlightModelQuantizer(ModelQuantizer):
    def __init__(self, config: Config) -> None:
        self.pytorchlight_quant_config = PytorchlightConfig.create_from_quark_config(config)
        self.config = config
        return

    def quantize_model(
        self,
        model: nn.Module,
        dataloader: Optional[Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]]]] = None
    ) -> nn.Module:

        quant_config = self.pytorchlight_quant_config
        q_model = auto_replace_to_light(model, mapping=quant_config.mapping, in_place=False)
        export_file_path = "pytorch-light-calib_value_export"
        if quant_config.brecq is not None:
            q_model = prepare(q_model, enable_optimizer_name=[quant_config.calibrate_method])
            model_eval(q_model, dataloader)
            export_calib_value(q_model, export_file_path)
            q_model = prepare(q_model, enable_optimizer_name=[])

            quant_configs['quant_round_mode.weight'] = 'ADAROUND'
            quant_configs['quant_optimizer_mode.global'] = {
                'WeightINTQuantizer': ['ROUNDOptimizer']
            }

            q_model = auto_replace_to_light(model, mapping=quant_config.mapping, in_place=False)
            load_calib_value(q_model, export_file_path)
            quant_config.brecq['brecq_params_config']['data_loader'] = dataloader
            quant_config.brecq['brecq_params_config']['device_for_optim'] = next(model.parameters()).device
            q_model.to(model.device)
            pytorchlight_BRECQ(model, q_model, quant_config.brecq['brecq_params_config'])
            return q_model

        if dataloader is not None:
            q_model = prepare(q_model, enable_optimizer_name=[quant_config.calibrate_method])
            model_eval(q_model, dataloader)
            q_model = prepare(q_model, enable_optimizer_name=[])
        return q_model

    def freeze(self, model):
        return model

class PytorchlightModelExporter(ModelExporter):
    def export_onnx_model(self, model: nn.Module, input_args: Union[torch.Tensor, Tuple[float]]) -> None:

        ONNXQDQManager.onnx_passes = []
        ONNXQDQManager.export(model, input_args, "./pytorch-light-export.onnx", do_constant_folding=False)
        return

def model_eval(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    n_samples = len(dataloader)
    iterator = iter(dataloader)

    for i in tqdm(range(n_samples)):
        batch = next(iterator)[0].to(device)
        with torch.no_grad():
            model(batch)
    model.config.use_cache = use_cache
