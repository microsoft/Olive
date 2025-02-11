from huggingface_hub import model_info
from diffusers import UNet2DConditionModel, UNetMotionModel, MotionAdapter
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from olive.data.registry import Registry
import os
import numpy as np
from pathlib import Path
import pickle


step = 2
unet_sample_size = 64
cross_attention_dim = 768
num_frames = 16
data_dir = Path("quantize_data")
data_num = 10

def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model") or model_name

def unet_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"

    adapter = MotionAdapter()
    adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt)))
    model = UNetMotionModel.from_unet2d(model, adapter)
    return model

def unet_inputs(batch_size, torch_dtype):
    inputs = {
        "sample": torch.rand((batch_size, 4, num_frames, unet_sample_size, unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batch_size,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batch_size * num_frames, 77, cross_attention_dim), dtype=torch_dtype),
    }
    return inputs

def unet_conversion_inputs(model=None):
    return tuple(unet_inputs(1, torch.float32).values())

# Quantization data

class BaseDataLoader:
    def __init__(self, total):
        self.data = []
        self.total = total
        self.data_folders = [data_dir / f.name for f in os.scandir(data_dir) if f.is_dir()]

    def __getitem__(self, idx):
        print("getitem: " + str(idx))
        if idx >= len(self.data) or idx >= self.total: return None
        return self.data[idx]
    
class UnetGeneratedDataLoader(BaseDataLoader):
    def __init__(self, total):
        super().__init__(total)
        
        for folder in self.data_folders:
            i = 0
            while True:
                f = folder / f'{i}_unet_input.bin'
                if not os.path.exists(f): break

                with open(f, 'rb') as file:
                    loaded_data = pickle.load(file)

                loaded_data = {key: torch.from_numpy(value) for key, value in loaded_data.items()}
                self.data.append(loaded_data)
                i += 1

@Registry.register_dataloader()
def unet_quantize_data_loader(dataset, batch_size, *args, **kwargs):
    return UnetGeneratedDataLoader(data_num)