#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import logging
import sys
from trainer import Trainer
from yolo_x_tiny_exp import Exp



def make_parser():
    parser = argparse.ArgumentParser("YOLOX quantization parser")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="pre traincheckpoint file")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument('--random_size_range', type=int, default=None, help='random_size')
    parser.add_argument("--experiment_name", type=str, default="0", help="exp name")
    parser.add_argument('--data_dir', default=None, help='Data set directory.')
    parser.add_argument("--min_lr_ratio", type=float, default=0.01, help="batch size")
    parser.add_argument("--ema_decay", type=float, default=0.9995, help="ema decay reate.")
    parser.add_argument('--output_dir', default='./YOLOX_outputs', help='Experiments results save path.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers to be used.')
    parser.add_argument('--multiscale_range', default=5, type=int, help='multiscale_range.')
    parser.add_argument("--start_epoch", type=int, default=280, help="batch size")
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        stream=sys.stdout)
    args = make_parser().parse_args()
    exp = Exp(args)
    trainer = Trainer(exp, args)
    trainer.quant_prepare()
    trainer.ptq()
    # evalue the PTQ results
    *_, summary = trainer.evaluator.evaluate(trainer.model)
    trainer.qat()
    # evalue the QAT results
    *_, summary = trainer.evaluator.evaluate(trainer.model)
    trainer.export_2_onnx_model()
