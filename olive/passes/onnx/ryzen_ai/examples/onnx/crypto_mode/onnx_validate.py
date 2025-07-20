#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
""" ONNX-runtime validation script

This script was created to verify accuracy and performance of exported ONNX
models running with the onnxruntime. It utilizes the PyTorch dataloader/processing
pipeline for a fair comparison against the originals.

Copyright 2020 Ross Wightman
"""
import os
import time
import argparse
import numpy as np

import torch
import torchvision
from torchvision import transforms
import onnxruntime

parser = argparse.ArgumentParser(description='ONNX Validation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--onnx-input',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-float',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-quant',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to onnx model/weights file')
parser.add_argument('--onnx-output-opt',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to output optimized onnx graph')
parser.add_argument('--profile',
                    action='store_true',
                    default=False,
                    help='Enable profiler output.')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=100,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 100)')
parser.add_argument('--print-freq',
                    '-p',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 100)')
parser.add_argument('--gpu',
                    action='store_true',
                    default=False,
                    help='Enable profiler output.')


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


def load_loader(data_dir, batch_size, workers):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(data_dir, data_transform)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers,
                                              pin_memory=True)
    return data_loader


def accuracy_np(output, target):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5],
                          target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


def evaluate(onnx_model_path, sess_options, providers, data_loader, print_freq):
    session = onnxruntime.InferenceSession(onnx_model_path,
                                           sess_options,
                                           providers=providers)
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
            print(
                f'Test: [{i}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {input.size(0) / batch_time.avg:.3f}/s, '
                f'{100 * batch_time.avg / input.size(0):.3f} ms/sample) \t'
                f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

    return top1, top5


def main():
    args = parser.parse_args()
    args.gpu_id = 0

    # Set graph optimization level
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.profile:
        sess_options.enable_profiling = True
    if args.onnx_output_opt:
        sess_options.optimized_model_filepath = args.onnx_output_opt
    if args.gpu:
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    if args.onnx_input:
        val_loader = load_loader(args.data, args.batch_size, args.workers)
        f_top1, f_top5 = evaluate(args.onnx_input, sess_options, providers,
                                  val_loader, args.print_freq)
        print(
            f' * Prec@1 {f_top1.avg:.3f} ({100 - f_top1.avg:.3f}) Prec@5 {f_top5.avg:.3f} ({100. - f_top5.avg:.3f})'
        )
    elif args.onnx_float and args.onnx_quant:
        val_loader = load_loader(args.data, args.batch_size, args.workers)
        f_top1, f_top5 = evaluate(args.onnx_float, sess_options, providers,
                                  val_loader, args.print_freq)
        f_top1 = format(f_top1.avg, '.2f')
        f_top5 = format(f_top5.avg, '.2f')

        q_top1, q_top5 = evaluate(args.onnx_quant, sess_options, providers,
                                  val_loader, args.print_freq)
        q_top1 = format(q_top1.avg, '.2f')
        q_top5 = format(q_top5.avg, '.2f')

        f_size = format(os.path.getsize(args.onnx_float) / (1024 * 1024), '.2f')
        q_size = format(os.path.getsize(args.onnx_quant) / (1024 * 1024), '.2f')
        """
        --------------------------------------------------------
        |             | float model    | quantized model |
        --------------------------------------------------------
        | ****        | ****           | ****             |
        --------------------------------------------------------
        | Model Size  | ****           | ****             |
        --------------------------------------------------------
        """
        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table()
        table.add_column('')
        table.add_column('Float Model')
        table.add_column('Quantized Model', style='bold green1')

        table.add_row("Model", args.onnx_float, args.onnx_quant)
        table.add_row("Model Size", str(f_size) + ' MB', str(q_size) + ' MB')
        table.add_row("Prec@1", str(f_top1) + ' %', str(q_top1) + ' %')
        table.add_row("Prec@5", str(f_top5) + ' %', str(q_top5) + ' %')

        console.print(table)

    else:
        print(
            "Please specify both model-float and model-quant or model-input for evaluation."
        )


if __name__ == '__main__':
    main()
