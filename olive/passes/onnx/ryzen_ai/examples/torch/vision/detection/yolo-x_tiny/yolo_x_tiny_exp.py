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

import argparse
import pprint
import random
from tabulate import tabulate
import torch
from utils import LRScheduler
from evaluators import COCOEvaluator
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d
from models import YOLOPAFPN, YOLOXHead, Quark_YOLOX, YOLOX
from data import COCODataset, TrainTransform, ValTransform, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection, worker_init_reset_seed
import torch.nn as nn


class Exp():
    """Basic class for yolo-x experiment."""

    def __init__(self, args: argparse.Namespace = None):
        self.seed = None
        self.output_dir = args.output_dir
        self.print_interval = 10
        self.eval_interval = 1
        self.dataset = None
        # ----------------YOLO_X TINY  model config ---------
        # detect classes number of model
        self.num_classes = 80
        # factor of model depth
        self.depth = 0.33
        # factor of model width
        self.width = 0.375
        # activation name.
        self.act = "silu"

        # ---------------- dataloader config ----------------
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = args.workers
        self.input_size = (416, 416)  # (height, width)
        # Actual multiscale ranges: [416 - 5 * 32, 416 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        self.random_size = (10, 20) if args.random_size_range is None else (13 - args.random_size_range,
                                                                            13 + args.random_size_range)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = args.data_dir
        # name of annotation file for training
        self.train_ann = "instances_train2017.json"
        # name of annotation file for evaluation
        self.val_ann = "instances_val2017.json"

        # --------------- transform config -----------------
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)  # YOLO_X TINY
        # apply mixup aug or not
        self.enable_mixup = False
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config ---------------------
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 300
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = args.min_lr_ratio
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = 'yolo_x_tiny_' + args.experiment_name
        # -----------------  testing config ------------------
        # output image size during evaluation/test
        self.test_size = (416, 416)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65
        # ----------------- check the value -------------------
        self.check_exp_value()

    def get_model(self):
        # NOTE the eps is important as the original model is trained based on this param.
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        self.model = Quark_YOLOX(YOLOX(backbone, head))
        self.model.apply(init_yolo)
        self.model.train()
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """

        return COCODataset(data_dir=self.data_dir,
                           json_file=self.train_ann,
                           img_size=self.input_size,
                           preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                           cache=cache,
                           cache_type=cache_type)

    def get_data_loader(self, batch_size, no_aug=False, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """

        self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def random_resize(self):
        tensor = torch.LongTensor(2).cuda()
        size_factor = self.input_size[1] * 1.0 / self.input_size[0]
        if not hasattr(self, 'random_size'):
            min_size = int(self.input_size[0] / 32) - self.multiscale_range
            max_size = int(self.input_size[0] / 32) + self.multiscale_range
            self.random_size = (min_size, max_size)
        size = random.randint(*self.random_size)
        size = (int(32 * size), 32 * int(size * size_factor))
        tensor[0] = size[0]
        tensor[1] = size[1]

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(inputs, size=tsize, mode="bilinear", align_corners=False)
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
            # pg0 bn's weight: 74， pg1 conv's weight：83, pg2 bias 83

            if hasattr(self.model, "base_model") and isinstance(self.model.base_model, torch.fx.GraphModule):
                for k, v in self.model.named_modules():
                    if isinstance(v, QuantizedConvBatchNorm2d):
                        assert v.bias is None
                        if isinstance(v.bn.bias, nn.Parameter):
                            pg2.append(v.bn.bias)
                        if isinstance(v.bn.weight, nn.Parameter):
                            pg0.append(v.bn.weight)  # no decay
                        if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                            pg1.append(v.weight)  # apply decay
                    elif isinstance(v, QuantConv2d):
                        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                            pg2.append(v.bias)  # biases
                        if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                            pg1.append(v.weight)
            else:
                raise ValueError("The input model is not quantable format")

            optimizer = torch.optim.SGD(pg0, lr=lr, momentum=self.momentum, nesterov=True)
            optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay})
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_dataset(self, **kwargs):
        legacy = kwargs.get("legacy", False)
        return COCODataset(data_dir=self.data_dir,
                           json_file=self.val_ann,
                           name="val2017",
                           img_size=self.test_size,
                           preproc=ValTransform(legacy=legacy))

    def get_eval_loader(self, batch_size, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)
        sampler = torch.utils.data.SequentialSampler(valdataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, testdev=False, legacy=False):
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

    def eval(self, model, evaluator, return_outputs=False):
        return evaluator.evaluate(model, return_outputs=return_outputs)

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [(str(k), pprint.pformat(v)) for k, v in vars(self).items() if not k.startswith("_")]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def check_exp_value(self):
        h, w = self.input_size
        assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"
