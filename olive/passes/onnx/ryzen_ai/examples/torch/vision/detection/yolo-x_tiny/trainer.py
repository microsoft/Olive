# Copyright (c) Megvii Inc. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on YOLOX(https://github.com/Megvii-BaseDetection/YOLOX).
# Licensed under Apache License 2.0.
#
# Modifications copyright(c) 2025 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
import os
import itertools
import argparse
import time
import datetime
import shutil
import logging
import math
from copy import deepcopy
from data import DataPrefetcher
from utils import MeterBuffer, adjust_status, gpu_mem_usage, mem_usage
from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.quantization.graph.torch_utils import _enable_observer, _enable_fake_quant, _active_bn
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
from quark.torch.quantization.observer.observer import PerTensorPowOf2MinMSEObserver
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
from yolo_x_tiny_exp import Exp


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


class Trainer:

    def __init__(self, exp: Exp, args: argparse.Namespace):
        self.exp = exp
        self.exp.eval_interval = 1
        self.args = args
        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.local_rank = 0
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt
        # data/dataloader related attr
        self.data_type = torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0
        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, exp.exp_name)
        self.start_epoch = args.start_epoch
        os.makedirs(self.file_name, exist_ok=True)
        self.quantizer = None

    def quant_prepare(self):
        logging.info("args: {}".format(self.args))
        logging.info("exp value:\n{}".format(self.exp))
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        model.to(self.device)
        model = self.load_pretrain_weight(model)
        self.model = model
        self.evaluator = self.exp.get_evaluator(batch_size=int(self.args.batch_size / 2))
        return

    def ptq(self):
        # NOTE as we select the  PerTensorPowOf2MinMSEObserver, this observer is very time cosuming,
        # so we se the calib data mini-batch to 1, as 1 can reach the resired PTQ result.
        calib_data = [x[0].to(self.device) for x in list(itertools.islice(self.evaluator.dataloader, 1))]
        dummy_input = torch.randn(1, 3, *self.exp.input_size).to(self.device)
        self.model = self.model.eval()

        # step 1 get the traced model that in torch.fx.GraphModule format
        # We only trace part of the model, code for loss computation, detection head decode are not need to trace.
        graph_model = torch.export.export_for_training(self.model.base_model, (dummy_input, )).module()
        graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
        self.model.base_model = graph_model

        # step 2 config the quant config
        INT8_PER_WEIGHT_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                       qscheme=QSchemeType.per_tensor,
                                                       observer_cls=PerTensorPowOf2MinMSEObserver,
                                                       symmetric=True,
                                                       scale_type=ScaleType.float,
                                                       round_method=RoundType.half_even,
                                                       is_dynamic=False)

        INT8_PER_ACTIVTION_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.uint8,
                                                          qscheme=QSchemeType.per_tensor,
                                                          observer_cls=PerTensorPowOf2MinMSEObserver,
                                                          symmetric=True,
                                                          scale_type=ScaleType.float,
                                                          round_method=RoundType.half_even,
                                                          is_dynamic=False)

        # quant config
        quant_config_0 = QuantizationConfig(weight=INT8_PER_WEIGHT_TENSOR_SPEC,
                                            input_tensors=INT8_PER_ACTIVTION_TENSOR_SPEC,
                                            output_tensors=INT8_PER_ACTIVTION_TENSOR_SPEC,
                                            bias=INT8_PER_WEIGHT_TENSOR_SPEC)
        quant_config_1 = QuantizationConfig(weight=INT8_PER_WEIGHT_TENSOR_SPEC,
                                            input_tensors=INT8_PER_WEIGHT_TENSOR_SPEC,
                                            output_tensors=INT8_PER_WEIGHT_TENSOR_SPEC,
                                            bias=INT8_PER_WEIGHT_TENSOR_SPEC)
        quant_config = Config(global_quant_config=quant_config_1, quant_mode=QuantizationMode.fx_graph_mode)
        self.quantizer = ModelQuantizer(quant_config)
        quantized_model = self.quantizer.quantize_model(graph_model, calib_data)
        if hasattr(self.model, "base_model"):
            self.model.base_model = quantized_model
        self.model = self.model.train()
        return

    def qat(self):
        #  ------------ prepare the training data, lr scheduler, optimizer--------
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(batch_size=self.args.batch_size,
                                                     no_aug=self.no_aug,
                                                     cache_img=None)
        logging.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)

        self.max_iter = len(self.train_loader)
        self.lr_scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter)
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        #  ------ using ema for better coverage ---
        if self.use_model_ema:
            self.ema_model = ModelEMA(self.model, self.args.ema_decay)  # 0.9995
            self.ema_model.updates = self.max_iter * self.start_epoch

        #  ----------- train the model -----------
        logging.info("Training start...")
        logging.info("\n{}".format(self.model))
        for self.epoch in range(self.start_epoch, self.max_epoch):
            #  ------ before training
            logging.info("---> start train epoch{}".format(self.epoch + 1))
            if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
                logging.info("--->No mosaic aug now!")
                self.train_loader.close_mosaic()
                logging.info("--->Add additional L1 loss now!")
                self.model.use_l1 = True
            #  ----- training
            self.train_in_iter()
            #  ----- post training
            if (self.epoch + 1) % self.exp.eval_interval == 0:
                self.evaluate_and_save_model()
        logging.info("Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100))
        return

    def _quant_train_mode_adjust(self, training=True):
        # during the training
        # 1. Close the observer as we have performed PTQ, the fluctuations of scale will make it hard for the network to converge.
        # 2. During the training, enable the fake quant to
        _enable_observer(self.model, not training)
        _enable_fake_quant(self.model, training)
        _active_bn(self.model, False)

    def train_in_iter(self):
        self.model = self.model.train()
        self._quant_train_mode_adjust(training=True)
        for self.iter in range(self.max_iter):
            iter_start_time = time.time()
            inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model(inps, targets)

            loss = outputs["total_loss"]

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.use_model_ema:
                self.ema_model.update(self.model)

            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            iter_end_time = time.time()
            self.meter.update(iter_time=iter_end_time - iter_start_time,
                              data_time=data_end_time - iter_start_time,
                              lr=lr,
                              **outputs)

            #  1.log information 2 reset setting of resize
            if (self.iter + 1) % self.exp.print_interval == 0:
                left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
                eta_seconds = self.meter["iter_time"].global_avg * left_iters
                eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

                progress_str = "epoch: {}/{}, iter: {}/{}".format(self.epoch + 1, self.max_epoch, self.iter + 1,
                                                                  self.max_iter)
                loss_meter = self.meter.get_filtered_meter("loss")
                loss_str = ", ".join(["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()])

                time_meter = self.meter.get_filtered_meter("time")
                time_str = ", ".join(["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()])

                mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

                logging.info("{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                ) + (", size: {:d}, {}".format(self.input_size[0], eta_str)))
                self.meter.clear_meters()

            if (self.progress_in_iter + 1) % 10 == 0:
                self.input_size = self.exp.random_resize()

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def load_pretrain_weight(self, model):
        if self.args.ckpt is not None:
            logging.info("loading pre-trained checkpoint")
            ckpt_file = self.args.ckpt
            ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
            model.base_model.load_state_dict(ckpt)
        return model

    def evaluate_and_save_model(self):
        # using ema model weight
        evalmodel = self.model  # NOTE haoliang
        evalmodel.load_state_dict(self.ema_model.ema.state_dict())
        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(evalmodel, self.evaluator, return_outputs=True)

        print("Epoch: {} eval ap50_95: {}".format(self.epoch, ap50_95))
        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)
        logging.info("\n" + summary)

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        save_model = self.ema_model.ema if self.use_model_ema else self.model
        logging.info("Save weights to {}".format(self.file_name))
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_ap": self.best_ap,
            "curr_ap": ap,
        }

        if not os.path.exists(self.file_name):
            os.makedirs(self.file_name)
        filename = os.path.join(self.file_name, ckpt_name + "_ckpt.pth")
        torch.save(ckpt_state, filename)
        if update_best_ckpt:
            best_filename = os.path.join(self.file_name, "best_ckpt.pth")
            shutil.copyfile(filename, best_filename)

    def export_2_onnx_model(self):
        # Freeze model and do post-quant optimization to meet hardware(NPU) compile requirements
        freezeded_model = self.quantizer.freeze(self.model.base_model.eval())
        self.model.base_model = freezeded_model
        config = ExporterConfig(json_export_config=JsonExporterConfig())
        exporter = ModelExporter(config=config, export_dir=self.file_name)
        # NOTE for NPU compile, it is better using batch-size = 1 for better compliance
        example_inputs = (torch.rand(1, 3, 416, 416).to(self.device), )
        exporter.export_onnx_model(self.model, example_inputs[0])
        # For better visualization, user can use simplify tool
        # from onnxsim import simplify
        # import onnx
        # quant_model = onnx.load("./***/quark_model.onnx")
        # model_simp, check = simplify(quant_model)
        # onnx.save_model(model_simp, "./sample_quark_model.onnx")
