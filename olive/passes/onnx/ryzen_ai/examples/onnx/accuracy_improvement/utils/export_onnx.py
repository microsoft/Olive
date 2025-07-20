#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import sys
import timm
import torch

model_name = sys.argv[1]

model = timm.create_model(model_name, pretrained=True)
model = model.eval()
device = torch.device("cpu")

data_config = timm.data.resolve_model_data_config(
    model=model,
    use_test_size=True,
)

batch_size = 1
torch.manual_seed(42)
dummy_input = torch.randn((batch_size, ) + tuple(data_config['input_size'])).to(device)

torch.onnx.export(model,
                  dummy_input,
                  "models/" + model_name + ".onnx",
                  export_params=True,
                  do_constant_folding=True,
                  opset_version=17,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={
                      'input': {
                          0: 'batch_size'
                      },
                      'output': {
                          0: 'batch_size'
                      }
                  },
                  verbose=True)
print("Onnx model is saved at models/" + model_name + ".onnx")
