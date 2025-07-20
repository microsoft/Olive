#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import time
import timm
import torch
import argparse
import torchvision
import onnxruntime
import numpy as np
from typing import Tuple
from argparse import Namespace
from torchvision import transforms
from timm.models import create_model
from timm.data import resolve_data_config
from quark.onnx.operators.custom_ops import get_library_path

def export_onnx_model(model_name: str) -> Tuple[str, str]:
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

    os.makedirs("models", exist_ok=True)

    input_model_path = "models/" + model_name + ".onnx"

    torch.onnx.export(model,
                      dummy_input,
                      input_model_path,
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
    print(f"ONNX model has been exported successfully at {input_model_path}.")
    return input_model_path

def load_loader(model_name, data_dir, batch_size, workers):
    timm_model = create_model(model_name, pretrained=False,)
    data_config = resolve_data_config(model=timm_model, use_test_size=True)
    crop_pct = data_config['crop_pct']
    input_size = data_config['input_size']
    width = input_size[-1]
    data_transform = transforms.Compose([
        transforms.Resize(int(width / crop_pct)),
        transforms.CenterCrop(width),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(data_dir, data_transform)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=True)
    return data_loader

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_np(output, target):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5], target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


def evaluate(onnx_model_path, sess_options, providers, data_loader, print_freq):
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)
    input_name = session.get_inputs()[0].name

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # run the net and return prediction
        output = session.run([], {input_name: input.data.numpy()})
        output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy_np(output, target.numpy())
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(f'Test: [{i}/{len(data_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {input.size(0) / batch_time.avg:.3f}/s, '
                  f'{100 * batch_time.avg / input.size(0):.3f} ms/sample) \t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

    return top1, top5

def evaluate_quantized_timm_model(model_name: str, input_model_path: str, evaluation_data_path: str, use_gpu: bool) -> None:
    args.gpu_id = 0

    # Set graph optimization level
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if use_gpu:
        providers = ['CUDAExecutionProvider']
        sess_options.register_custom_ops_library(get_library_path("CUDA"))
    else:
        providers = ['CPUExecutionProvider']
        sess_options.register_custom_ops_library(get_library_path("CPU"))
    val_loader = load_loader(model_name, evaluation_data_path, 100, 1)
    f_top1, f_top5 = evaluate(input_model_path, sess_options, providers, val_loader, 1)
    print(f' * Prec@1 {f_top1.avg:.3f} ({100 - f_top1.avg:.3f}) Prec@5 {f_top5.avg:.3f} ({100. - f_top5.avg:.3f})')

def main(args: argparse.Namespace) -> None:
    input_model_path = export_onnx_model(args.model_name)
    evaluate_quantized_timm_model(args.model_name, input_model_path, args.eval_data_path, args.gpu)

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", help="Specify the input model name to be quantized", required=True)
    parser.add_argument("--eval_data_path",
                        help="The path of the folder for evaluation",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument('--gpu', action='store_true', default=False, help='Whether use onnxruntime-gpu to infer.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
