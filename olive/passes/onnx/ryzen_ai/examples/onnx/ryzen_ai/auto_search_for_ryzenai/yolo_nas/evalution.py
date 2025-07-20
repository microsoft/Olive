#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import inspect
import os
import typing
import warnings
from copy import deepcopy
from typing import Union, Tuple, Mapping, Dict, Any, List, Optional

import hydra
import numpy as np
import torch
import torch.cuda
import torch.nn
import torchmetrics
from omegaconf import DictConfig, OmegaConf

from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MetricCollection, Metric
from tqdm import tqdm

from super_gradients import is_distributed
from super_gradients.common.environment.checkpoints_dir_utils import (
    generate_run_id,
    get_latest_run_id,
    validate_run_id,
    get_checkpoints_dir_path,
)
from super_gradients.module_interfaces import HasPreprocessingParams, HasPredict
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches

from super_gradients.training.utils.sg_trainer_utils import get_callable_param_names
from super_gradients.training.utils.callbacks.callbacks import create_lr_scheduler_callback, LRSchedulerCallback
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.sg_loggers.abstract_sg_logger import AbstractSGLogger
from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.data_types.enum import MultiGPUMode, StrictLoad, EvaluationType
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.callbacks_factory import CallbacksFactory
from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.losses_factory import LossesFactory
from super_gradients.common.factories.metrics_factory import MetricsFactory
from super_gradients.common.environment.package_utils import get_installed_packages
from super_gradients.common.environment.cfg_utils import maybe_instantiate_test_loaders

from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.training.datasets.samplers import RepeatAugSampler
from super_gradients.common.exceptions.sg_trainer_exceptions import UnsupportedOptimizerFormat
from super_gradients.training.metrics.metric_utils import (
    get_metrics_titles,
    get_metrics_results_tuple,
    get_logging_values,
    get_metrics_dict,
    get_train_loop_description_dict,
)
from super_gradients.training.models import SgModule, get_model_name
from super_gradients.common.registry.registry import ARCHITECTURES, SG_LOGGERS
from super_gradients.training.pretrained_models import PRETRAINED_NUM_CLASSES
from super_gradients.training.utils import sg_trainer_utils, get_param, torch_version_is_greater_or_equal
from super_gradients.training.utils.distributed_training_utils import (
    MultiGPUModeAutocastWrapper,
    reduce_results_tuple_for_ddp,
    compute_precise_bn_stats,
    setup_device,
    get_gpu_mem_utilization,
    wait_for_the_master,
    DDPNotSetupException,
)
from super_gradients.common.environment.ddp_utils import (
    get_local_rank,
    require_ddp_setup,
    is_ddp_subprocess,
    get_world_size,
    broadcast_from_master,
)
from super_gradients.training.utils.ema import ModelEMA
from super_gradients.training.utils.optimizer_utils import build_optimizer, get_initial_lr_from_optimizer
from super_gradients.training.utils.sg_trainer_utils import MonitoredValue, log_main_training_params
from super_gradients.training.utils.utils import fuzzy_idx_in_list, unwrap_model
from super_gradients.training.utils.weight_averaging_utils import ModelWeightAveraging
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.utils import random_seed
from super_gradients.training.utils.checkpoint_utils import (
    read_ckpt_state_dict,
    load_checkpoint_to_model,
    load_pretrained_weights,
    get_scheduler_state,
)
from super_gradients.training.datasets.datasets_utils import DatasetStatisticsTensorboardLogger
from super_gradients.training.utils.callbacks import (
    CallbackHandler,
    Phase,
    PhaseContext,
    MetricsUpdateCallback,
    LRCallbackBase,
)
from super_gradients.common.registry.registry import LR_WARMUP_CLS_DICT
from super_gradients.common.environment.device_utils import device_config
from super_gradients.training.utils import HpmStruct
from super_gradients.common.environment.cfg_utils import load_experiment_cfg, add_params_to_cfg, load_recipe
from super_gradients.common.factories.pre_launch_callbacks_factory import PreLaunchCallbacksFactory
from super_gradients.training.params import TrainingParams
from super_gradients.module_interfaces import ExportableObjectDetectionModel, SupportsInputShapeCheck
from super_gradients.conversion import ExportQuantizationMode

logger = get_logger(__name__)


try:
    from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
    from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
    from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

    _imported_pytorch_quantization_failure = None

except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug("Failed to import pytorch_quantization:")
    logger.debug(import_err)
    _imported_pytorch_quantization_failure = import_err


class Trainer:

    def __init__(self, experiment_name: str, device: Optional[str] = None, multi_gpu: Union[MultiGPUMode, str] = None, ckpt_root_dir: Optional[str] = None):

        if device is not None or multi_gpu is not None:
            raise KeyError(
                "Trainer does not accept anymore 'device' and 'multi_gpu' as argument. "
                "Both should instead be passed to "
                "super_gradients.setup_device(device=..., multi_gpu=..., num_gpus=...)"
            )

        if require_ddp_setup():
            raise DDPNotSetupException()

        self.net, self.architecture, self.arch_params, self.dataset_interface = None, None, None, None
        self.train_loader, self.valid_loader, self.test_loaders = None, None, {}
        self.ema = None
        self.ema_model = None
        self.sg_logger = None
        self.update_param_groups = None
        self.criterion = None
        self.training_params = None
        self.scaler = None
        self.phase_callbacks = None
        self.checkpoint_params = None
        self.pre_prediction_callback = None

        self.half_precision = False
        self.load_backbone = False
        self.load_weights_only = False
        self.ddp_silent_mode = is_ddp_subprocess()

        self.model_weight_averaging = None
        self.average_model_checkpoint_filename = "average_model.pth"
        self.start_epoch = 0
        self.best_metric = np.inf
        self.load_ema_as_net = False

        self._first_backward = True

        self.loss_logging_items_names = None
        self.train_metrics: Optional[MetricCollection] = None
        self.valid_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None
        self.greater_metric_to_watch_is_better = None
        self.metric_to_watch = None
        self.greater_train_metrics_is_better: Dict[str, bool] = {}
        self.greater_valid_metrics_is_better: Dict[str, bool] = {}

        self.ckpt_root_dir = ckpt_root_dir
        self.experiment_name = experiment_name
        self.checkpoints_dir_path = None
        self.load_checkpoint = False
        self.ckpt_best_name = "ckpt_best.pth"

        self.phase_callback_handler: CallbackHandler = None

        default_results_titles = ["Train Loss", "Train Acc", "Train Top5", "Valid Loss", "Valid Acc", "Valid Top5"]

        self.results_titles = default_results_titles

        default_train_metrics, default_valid_metrics = MetricCollection([Accuracy(), Top5()]), MetricCollection([Accuracy(), Top5()])

        self.train_metrics, self.valid_metrics = default_train_metrics, default_valid_metrics

        self.train_monitored_values = {}
        self.valid_monitored_values = {}
        self.test_monitored_values = {}
        self.max_train_batches = None
        self.max_valid_batches = None

        self._epoch_start_logging_values = {}
        self._torch_lr_scheduler = None

    @property
    def device(self) -> str:
        return device_config.device

    @classmethod
    def train_from_config(cls, cfg: Union[DictConfig, dict]) -> Tuple[nn.Module, Tuple]:
        setup_device(
            device=core_utils.get_param(cfg, "device"),
            multi_gpu=core_utils.get_param(cfg, "multi_gpu"),
            num_gpus=core_utils.get_param(cfg, "num_gpus"),
        )

        cfg = hydra.utils.instantiate(cfg)
        cfg = cls._trigger_cfg_modifying_callbacks(cfg)

        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir)

        model = models.get(
            model_name=cfg.architecture,
            num_classes=cfg.arch_params.num_classes,
            arch_params=cfg.arch_params,
            strict_load=cfg.checkpoint_params.strict_load,
            pretrained_weights=cfg.checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.checkpoint_params.checkpoint_path,
            load_backbone=cfg.checkpoint_params.load_backbone,
            checkpoint_num_classes=get_param(cfg.checkpoint_params, "checkpoint_num_classes"),
            num_input_channels=get_param(cfg.arch_params, "num_input_channels"),
        )


        train_dataloader = dataloaders.get(
            name=get_param(cfg, "train_dataloader"),
            dataset_params=cfg.dataset_params.train_dataset_params,
            dataloader_params=cfg.dataset_params.train_dataloader_params,
        )

        val_dataloader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=cfg.dataset_params.val_dataset_params,
            dataloader_params=cfg.dataset_params.val_dataloader_params,
        )

        test_loaders = maybe_instantiate_test_loaders(cfg)

        recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}
        res = trainer.train(
            model=model,
            train_loader=train_dataloader,
            valid_loader=val_dataloader,
            test_loaders=test_loaders,
            training_params=cfg.training_hyperparams,
            additional_configs_to_log=recipe_logged_cfg,
        )

        return model, res

    @classmethod
    def _trigger_cfg_modifying_callbacks(cls, cfg):
        pre_launch_cbs = get_param(cfg, "pre_launch_callbacks_list", list())
        pre_launch_cbs = ListFactory(PreLaunchCallbacksFactory()).get(pre_launch_cbs)
        for plcb in pre_launch_cbs:
            cfg = plcb(cfg)
        return cfg

    @classmethod
    def resume_experiment(cls, experiment_name: str, ckpt_root_dir: Optional[str] = None, run_id: Optional[str] = None) -> Tuple[nn.Module, Tuple]:
        """
        Resume a training that was run using our recipes.

        :param experiment_name:     Name of the experiment to resume
        :param ckpt_root_dir:       Directory including the checkpoints
        :param run_id:              Optional. Run id of the experiment. If None, the most recent run will be loaded.
        :return:                    The config that was used for that experiment
        """
        logger.info("Resume training using the checkpoint recipe, ignoring the current recipe")

        if run_id is None:
            run_id = get_latest_run_id(checkpoints_root_dir=ckpt_root_dir, experiment_name=experiment_name)

        cfg = load_experiment_cfg(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name, run_id=run_id)

        add_params_to_cfg(cfg, params=["training_hyperparams.resume=True"])
        if run_id:
            add_params_to_cfg(cfg, params=[f"training_hyperparams.run_id={run_id}"])
        return cls.train_from_config(cfg)

    @classmethod
    def evaluate_from_recipe(cls, cfg: DictConfig) -> Tuple[nn.Module, Tuple]:

        setup_device(
            device=core_utils.get_param(cfg, "device"),
            multi_gpu=core_utils.get_param(cfg, "multi_gpu"),
            num_gpus=core_utils.get_param(cfg, "num_gpus"),
        )


        cfg = hydra.utils.instantiate(cfg)

        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir)

        val_dataloader = dataloaders.get(
            name=cfg.val_dataloader, dataset_params=cfg.dataset_params.val_dataset_params, dataloader_params=cfg.dataset_params.val_dataloader_params
        )

        if cfg.checkpoint_params.checkpoint_path is None:
            logger.info(
                "`checkpoint_params.checkpoint_path` was not provided. The recipe will be evaluated using checkpoints_dir.training_hyperparams.ckpt_name"
            )

            eval_run_id = core_utils.get_param(cfg, "training_hyperparams.run_id", None)
            if eval_run_id is None:
                logger.info("`training_hyperparams.run_id` was not provided. Evaluating the latest run.")
                eval_run_id = get_latest_run_id(checkpoints_root_dir=cfg.ckpt_root_dir, experiment_name=cfg.experiment_name)

            checkpoints_dir = get_checkpoints_dir_path(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir, run_id=eval_run_id)
            checkpoint_path = os.path.join(checkpoints_dir, cfg.training_hyperparams.ckpt_name)
            if os.path.exists(checkpoint_path):
                cfg.checkpoint_params.checkpoint_path = checkpoint_path

        logger.info(f"Evaluating checkpoint: {cfg.checkpoint_params.checkpoint_path}")

        model = models.get(
            model_name=cfg.architecture,
            num_classes=cfg.arch_params.num_classes,
            arch_params=cfg.arch_params,
            strict_load=cfg.checkpoint_params.strict_load,
            pretrained_weights=cfg.checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.checkpoint_params.checkpoint_path,
            load_backbone=cfg.checkpoint_params.load_backbone,
            checkpoint_num_classes=get_param(cfg.checkpoint_params, "checkpoint_num_classes"),
            num_input_channels=get_param(cfg.arch_params, "num_input_channels"),
        )

        valid_metrics_dict = trainer.test(model=model, test_loader=val_dataloader, test_metrics_list=cfg.training_hyperparams.valid_metrics_list)

        results = ["Validate Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in valid_metrics_dict.items()]
        logger.info("\n".join(results))

        return model, valid_metrics_dict

    @classmethod
    def evaluate_checkpoint(
        cls,
        experiment_name: str,
        ckpt_name: str = "ckpt_latest.pth",
        ckpt_root_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:

        logger.info("Evaluate checkpoint")

        if run_id is None:
            run_id = get_latest_run_id(checkpoints_root_dir=ckpt_root_dir, experiment_name=experiment_name)

        cfg = load_experiment_cfg(ckpt_root_dir=ckpt_root_dir, experiment_name=experiment_name, run_id=run_id)

        add_params_to_cfg(cfg, params=["training_hyperparams.resume=True", f"ckpt_name={ckpt_name}"])
        cls.evaluate_from_recipe(cfg)

    def _net_to_device(self):

        self.net.to(device_config.device)

        sync_bn = core_utils.get_param(self.training_params, "sync_bn", default_val=False)
        if device_config.multi_gpu == MultiGPUMode.DATA_PARALLEL:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(device_config.num_gpus)))
        elif device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            if sync_bn:
                if not self.ddp_silent_mode:
                    logger.info("DDP - Using Sync Batch Norm... Training time will be affected accordingly")
                self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)

            local_rank = int(device_config.device.split(":")[1])
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    def _train_epoch(self, context: PhaseContext, silent_mode: bool = False) -> tuple:

        self.net.train()

        expected_iterations = len(self.train_loader) if self.max_train_batches is None else self.max_train_batches

        with tqdm(
            self.train_loader, total=expected_iterations, bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True, disable=silent_mode
        ) as progress_bar_train_loader:
            progress_bar_train_loader.set_description(f"Train epoch {context.epoch}")

            self._reset_metrics()

            self.train_metrics.to(device_config.device)
            loss_avg_meter = core_utils.utils.AverageMeter()

            context.update_context(loss_avg_meter=loss_avg_meter, metrics_compute_fn=self.train_metrics)

            for batch_idx, batch_items in enumerate(progress_bar_train_loader):
                if expected_iterations <= batch_idx:
                    break

                batch_items = core_utils.tensor_container_to_device(batch_items, device_config.device, non_blocking=True)
                inputs, targets, additional_batch_items = sg_trainer_utils.unpack_batch_items(batch_items)

                if self.pre_prediction_callback is not None:
                    inputs, targets = self.pre_prediction_callback(inputs, targets, batch_idx)

                context.update_context(
                    batch_idx=batch_idx, inputs=inputs, target=targets, additional_batch_items=additional_batch_items, **additional_batch_items
                )
                self.phase_callback_handler.on_train_batch_start(context)


                with autocast(enabled=self.training_params.mixed_precision):

                    outputs = self.net(inputs)

                    loss, loss_log_items = self._get_losses(outputs, targets)

                context.update_context(preds=outputs, loss_log_items=loss_log_items, loss_logging_items_names=self.loss_logging_items_names)
                self.phase_callback_handler.on_train_batch_loss_end(context)

                if not self.ddp_silent_mode and batch_idx == 0:
                    self._epoch_start_logging_values = self._get_epoch_start_logging_values()

                self._backward_step(loss, context.epoch, batch_idx, context)


                logging_values = loss_avg_meter.average + get_metrics_results_tuple(self.train_metrics)
                gpu_memory_utilization = get_gpu_mem_utilization() / 1e9 if torch.cuda.is_available() else 0


                pbar_message_dict = get_train_loop_description_dict(
                    logging_values, self.train_metrics, self.loss_logging_items_names, gpu_mem=gpu_memory_utilization
                )

                progress_bar_train_loader.set_postfix(**pbar_message_dict)
                self.phase_callback_handler.on_train_batch_end(context)

            self.train_monitored_values = sg_trainer_utils.update_monitored_values_dict(
                monitored_values_dict=self.train_monitored_values, new_values_dict=pbar_message_dict
            )

        return logging_values

    def _get_losses(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, tuple]:

        loss = self.criterion(outputs, targets)
        if isinstance(loss, tuple):
            loss, loss_logging_items = loss

        else:
            loss_logging_items = loss.unsqueeze(0).detach()

        if self.loss_logging_items_names is None or self._first_backward:
            self._init_loss_logging_names(loss_logging_items)
            if self.metric_to_watch:
                self._init_monitored_items()
            self._first_backward = False

        if len(loss_logging_items) != len(self.loss_logging_items_names):
            raise ValueError(
                "Loss output length must match loss_logging_items_names. Got "
                + str(len(loss_logging_items))
                + ", and "
                + str(len(self.loss_logging_items_names))
            )

        return loss, loss_logging_items

    def _init_monitored_items(self):

        for loss_name in self.loss_logging_items_names:
            self.train_monitored_values[loss_name] = MonitoredValue(name=loss_name, greater_is_better=False)
            self.valid_monitored_values[loss_name] = MonitoredValue(name=loss_name, greater_is_better=False)

        for metric_name in get_metrics_titles(self.train_metrics):
            self.train_monitored_values[metric_name] = MonitoredValue(name=metric_name, greater_is_better=self.greater_train_metrics_is_better.get(metric_name))

        for metric_name in get_metrics_titles(self.valid_metrics):
            self.valid_monitored_values[metric_name] = MonitoredValue(name=metric_name, greater_is_better=self.greater_valid_metrics_is_better.get(metric_name))

        for dataset_name in self.test_loaders.keys():
            for loss_name in self.loss_logging_items_names:
                loss_full_name = f"{dataset_name}:{loss_name}" if dataset_name else loss_name
                self.test_monitored_values[loss_full_name] = MonitoredValue(
                    name=f"{dataset_name}:{loss_name}",
                    greater_is_better=False,
                )
            for metric_name in get_metrics_titles(self.test_metrics):
                metric_full_name = f"{dataset_name}:{metric_name}" if dataset_name else metric_name
                self.test_monitored_values[metric_full_name] = MonitoredValue(
                    name=metric_full_name,
                    greater_is_better=self.greater_valid_metrics_is_better.get(metric_name),
                )


        metric_titles = self.loss_logging_items_names + get_metrics_titles(self.valid_metrics)
        try:
            metric_to_watch_idx = fuzzy_idx_in_list(self.metric_to_watch, metric_titles)
        except IndexError:
            raise ValueError(f"No match found for `metric_to_watch={self.metric_to_watch}`. Available metrics to monitor are: `{metric_titles}`.")

        metric_to_watch = metric_titles[metric_to_watch_idx]
        if metric_to_watch != self.metric_to_watch:
            logger.warning(
                f"No exact match found for `metric_to_watch={self.metric_to_watch}`. Available metrics to monitor are: `{metric_titles}`. \n"
                f"`metric_to_watch={metric_to_watch} will be used instead.`"
            )
            self.metric_to_watch = metric_to_watch

        if self.training_params.average_best_models:
            self.model_weight_averaging = ModelWeightAveraging(
                ckpt_dir=self.checkpoints_dir_path,
                greater_is_better=self.greater_metric_to_watch_is_better,
                metric_to_watch=self.metric_to_watch,
                load_checkpoint=self.load_checkpoint,
            )

    def _backward_step(self, loss: torch.Tensor, epoch: int, batch_idx: int, context: PhaseContext, *args, **kwargs) -> None:

        self.scaler.scale(loss).backward()
        self.phase_callback_handler.on_train_batch_backward_end(context)

        local_step = batch_idx + 1
        global_step = local_step + len(self.train_loader) * epoch
        total_steps = len(self.train_loader) * self.max_epochs

        if global_step % self.batch_accumulate == 0:
            self.phase_callback_handler.on_train_batch_gradient_step_start(context)

            if self.training_params.clip_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.training_params.clip_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()
            if self.ema:
                self.ema_model.update(self.net, step=global_step, total_steps=total_steps)

            self.phase_callback_handler.on_train_batch_gradient_step_end(context)

    def _save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer = None,
        epoch: int = None,
        train_metrics_dict: Optional[Dict[str, float]] = None,
        validation_results_dict: Optional[Dict[str, float]] = None,
        context: PhaseContext = None,
    ) -> None:

        if validation_results_dict is None:
            self.sg_logger.add_checkpoint(tag="ckpt_latest_weights_only.pth", state_dict={"net": self.net.state_dict()}, global_step=epoch)
            return

        curr_tracked_metric = float(validation_results_dict[self.metric_to_watch])

        valid_metrics_titles = get_metrics_titles(self.valid_metrics)

        all_metrics = {
            "tracked_metric_name": self.metric_to_watch,
            "valid": {metric_name: float(validation_results_dict[metric_name]) for metric_name in valid_metrics_titles},
        }

        if train_metrics_dict is not None:
            train_metrics_titles = get_metrics_titles(self.train_metrics)
            all_metrics["train"] = {metric_name: float(train_metrics_dict[metric_name]) for metric_name in train_metrics_titles}

        state = {
            "net": unwrap_model(self.net).state_dict(),
            "acc": curr_tracked_metric,
            "epoch": epoch,
            "metrics": all_metrics,
            "packages": get_installed_packages(),
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        if self.ema:
            state["ema_net"] = unwrap_model(self.ema_model.ema).state_dict()

        processing_params = self._get_preprocessing_from_valid_loader()
        if processing_params is not None:
            state["processing_params"] = processing_params

        if self._torch_lr_scheduler is not None:
            state["torch_scheduler_state_dict"] = get_scheduler_state(self._torch_lr_scheduler)


        self.sg_logger.add_checkpoint(tag="ckpt_latest.pth", state_dict=state, global_step=epoch)


        if epoch in self.training_params.save_ckpt_epoch_list:
            self.sg_logger.add_checkpoint(tag=f"ckpt_epoch_{epoch}.pth", state_dict=state, global_step=epoch)


        if (curr_tracked_metric > self.best_metric and self.greater_metric_to_watch_is_better) or (
            curr_tracked_metric < self.best_metric and not self.greater_metric_to_watch_is_better
        ):

            self.best_metric = curr_tracked_metric
            self.sg_logger.add_checkpoint(tag=self.ckpt_best_name, state_dict=state, global_step=epoch)


            self.phase_callback_handler.on_validation_end_best_epoch(context)
            logger.info("Best checkpoint overriden: validation " + self.metric_to_watch + ": " + str(curr_tracked_metric))

        if self.training_params.average_best_models:
            net_for_averaging = unwrap_model(self.ema_model.ema if self.ema else self.net)
            state["net"] = self.model_weight_averaging.get_average_model(net_for_averaging, validation_results_dict=validation_results_dict)

            for key_to_remove in ["optimizer_state_dict", "scaler_state_dict", "ema_net"]:
                _ = state.pop(key_to_remove, None)
            self.sg_logger.add_checkpoint(tag=self.average_model_checkpoint_filename, state_dict=state, global_step=epoch)

    def _prep_net_for_train(self) -> None:
        if self.arch_params is None:
            self._init_arch_params()

        if self.checkpoint_params is None:
            self.checkpoint_params = HpmStruct(load_checkpoint=self.training_params.resume)

        self._net_to_device()

        self.update_param_groups = hasattr(unwrap_model(self.net), "update_param_groups")

        if self.training_params.torch_compile:
            if torch_version_is_greater_or_equal(2, 0):
                logger.info("Using torch.compile feature. Compiling model. This may take a few minutes")
                self.net = torch.compile(self.net, **self.training_params.torch_compile_options)
                logger.info("Model compilation complete. Continuing training")
                if is_distributed():
                    torch.distributed.barrier()
            else:
                logger.warning(
                    "Your recipe has requested use of torch.compile. "
                    f"However torch.compile is not supported in this version of PyTorch ({torch.__version__}). "
                    "A Pytorch 2.0 or greater version is required. Ignoring torch_compile flag"
                )

    def _init_arch_params(self) -> None:
        default_arch_params = HpmStruct()
        arch_params = getattr(self.net, "arch_params", default_arch_params)
        self.arch_params = default_arch_params
        if arch_params is not None:
            self.arch_params.override(**arch_params.to_dict())

    def _should_run_validation_for_epoch(self, epoch: int) -> bool:

        is_run_val_freq_divisible = ((epoch + 1) % self.run_validation_freq) == 0
        is_last_epoch = (epoch + 1) == self.max_epochs
        is_in_checkpoint_list = (epoch + 1) in self.training_params.save_ckpt_epoch_list

        return is_run_val_freq_divisible or is_last_epoch or is_in_checkpoint_list

    def train(
        self,
        model: nn.Module,
        training_params: dict = None,
        train_loader: DataLoader = None,
        valid_loader: DataLoader = None,
        test_loaders: Dict[str, DataLoader] = None,
        additional_configs_to_log: Dict = None,
    ):

        global logger
        if training_params is None:
            training_params = dict()

        self.train_loader = train_loader if train_loader is not None else self.train_loader
        self.valid_loader = valid_loader if valid_loader is not None else self.valid_loader
        self.test_loaders = test_loaders if test_loaders is not None else {}

        if self.train_loader is None:
            raise ValueError("No `train_loader` found. Please provide a value for `train_loader`")

        if self.valid_loader is None:
            raise ValueError("No `valid_loader` found. Please provide a value for `valid_loader`")

        if self.test_loaders is not None and not isinstance(self.test_loaders, dict):
            raise ValueError("`test_loaders` must be a dictionary mapping dataset names to DataLoaders")

        if hasattr(self.train_loader, "batch_sampler") and self.train_loader.batch_sampler is not None:
            batch_size = self.train_loader.batch_sampler.batch_size
        else:
            batch_size = self.train_loader.batch_size

        if len(self.train_loader.dataset) % batch_size != 0 and not self.train_loader.drop_last:
            logger.warning("Train dataset size % batch_size != 0 and drop_last=False, this might result in smaller " "last batch.")

        if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:

            train_sampler = self.train_loader.batch_sampler.sampler if self.train_loader.batch_sampler is not None else self.train_loader.sampler
            if isinstance(train_sampler, SequentialSampler):
                raise ValueError(
                    "You are using a SequentialSampler on you training dataloader, while working on DDP. "
                    "This cancels the DDP benefits since it makes each process iterate through the entire dataset"
                )
            if not isinstance(train_sampler, (DistributedSampler, RepeatAugSampler)):
                logger.warning(
                    "The training sampler you are using might not support DDP. "
                    "If it doesnt, please use one of the following sampler: DistributedSampler, RepeatAugSampler"
                )
        self.training_params = TrainingParams()
        if isinstance(training_params, DictConfig):
            training_params = OmegaConf.to_container(training_params, resolve=True)
        self.training_params.override(**training_params)

        self.net = model

        self._prep_net_for_train()
        self._load_checkpoint_to_model()
        if not self.ddp_silent_mode:
            self._initialize_sg_logger_objects(additional_configs_to_log)


        random_seed(is_ddp=device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL, device=device_config.device, seed=self.training_params.seed)

        silent_mode = self.training_params.silent_mode or self.ddp_silent_mode

        self._set_train_metrics(train_metrics_list=self.training_params.train_metrics_list)
        self._set_valid_metrics(valid_metrics_list=self.training_params.valid_metrics_list)
        self.test_metrics = self.valid_metrics.clone()


        self.metric_to_watch = self.training_params.metric_to_watch
        self.greater_metric_to_watch_is_better = self.training_params.greater_metric_to_watch_is_better

        if isinstance(self.training_params.loss, str):
            self.criterion = LossesFactory().get({self.training_params.loss: self.training_params.criterion_params})

        elif isinstance(self.training_params.loss, Mapping):
            self.criterion = LossesFactory().get(self.training_params.loss)

        elif isinstance(self.training_params.loss, nn.Module):
            self.criterion = self.training_params.loss

        self.criterion.to(device_config.device)

        if self.training_params.torch_compile_loss:
            if torch_version_is_greater_or_equal(2, 0):
                logger.info("Using torch.compile feature. Compiling loss. This may take a few minutes")
                self.criterion = torch.compile(self.criterion, **self.training_params.torch_compile_options)
                logger.info("Loss compilation complete. Continuing training")
                if is_distributed():
                    torch.distributed.barrier()
            else:
                logger.warning(
                    "Your recipe has requested use of torch.compile. "
                    f"However torch.compile is not supported in this version of PyTorch ({torch.__version__}). "
                    "A Pytorch 2.0 or greater version is required. Ignoring torch_compile flag"
                )

        self.max_epochs = self.training_params.max_epochs

        self.ema = self.training_params.ema

        self.precise_bn = self.training_params.precise_bn
        self.precise_bn_batch_size = self.training_params.precise_bn_batch_size

        self.batch_accumulate = self.training_params.batch_accumulate
        num_batches = len(self.train_loader)

        if self.ema:
            self.ema_model = self._instantiate_ema_model(self.training_params.ema_params)
            self.ema_model.updates = self.start_epoch * num_batches // self.batch_accumulate
            if self.load_checkpoint:
                if "ema_net" in self.checkpoint.keys():
                    self.ema_model.ema.load_state_dict(self.checkpoint["ema_net"])
                else:
                    self.ema = False
                    logger.warning("[Warning] Checkpoint does not include EMA weights, continuing training without EMA.")

        self.run_validation_freq = self.training_params.run_validation_freq

        if self.max_epochs % self.run_validation_freq != 0:
            logger.warning(
                "max_epochs is not divisible by run_validation_freq. "
                "Please check the training parameters and ensure that run_validation_freq has been set correctly."
            )
        self.run_test_freq = self.training_params.run_test_freq

        timer = core_utils.Timer(device_config.device)

        self.lr_mode = self.training_params.lr_mode
        load_opt_params = self.training_params.load_opt_params

        self.phase_callbacks = self.training_params.phase_callbacks or []
        self.phase_callbacks = ListFactory(CallbacksFactory()).get(self.phase_callbacks)

        warmup_mode = self.training_params.warmup_mode
        warmup_callback_cls = None
        if isinstance(warmup_mode, str):
            from super_gradients.common.registry.registry import warn_if_deprecated

            warn_if_deprecated(warmup_mode, LR_WARMUP_CLS_DICT)

            warmup_callback_cls = LR_WARMUP_CLS_DICT[warmup_mode]
        elif isinstance(warmup_mode, type) and issubclass(warmup_mode, LRCallbackBase):
            warmup_callback_cls = warmup_mode
        elif warmup_mode is not None:
            pass
        else:
            raise RuntimeError("warmup_mode has to be either a name of a mode (str) or a subclass of PhaseCallback")

        if isinstance(self.training_params.optimizer, str) or (
            inspect.isclass(self.training_params.optimizer) and issubclass(self.training_params.optimizer, torch.optim.Optimizer)
        ):
            self.optimizer = build_optimizer(net=unwrap_model(self.net), lr=self.training_params.initial_lr, training_params=self.training_params)
        elif isinstance(self.training_params.optimizer, torch.optim.Optimizer):
            if self.training_params.initial_lr is not None:
                raise RuntimeError("An instantiated optimizer cannot be passed along initial_lr != None")
            self.optimizer = self.training_params.optimizer

            self.training_params.initial_lr = get_initial_lr_from_optimizer(self.optimizer)
        else:
            raise UnsupportedOptimizerFormat()

        if warmup_callback_cls is not None:
            self.phase_callbacks.append(
                warmup_callback_cls(
                    train_loader_len=len(self.train_loader),
                    net=self.net,
                    training_params=self.training_params,
                    update_param_groups=self.update_param_groups,
                    **self.training_params.to_dict(),
                )
            )

        self._add_metrics_update_callback(Phase.TRAIN_BATCH_END)
        self._add_metrics_update_callback(Phase.VALIDATION_BATCH_END)
        self._add_metrics_update_callback(Phase.TEST_BATCH_END)

        self.phase_callback_handler = CallbackHandler(callbacks=self.phase_callbacks)

        if not self.ddp_silent_mode:
            if self.training_params.dataset_statistics:
                dataset_statistics_logger = DatasetStatisticsTensorboardLogger(self.sg_logger)
                dataset_statistics_logger.analyze(
                    self.train_loader, all_classes=self.classes, title="Train-set", anchors=unwrap_model(self.net).arch_params.anchors
                )
                dataset_statistics_logger.analyze(self.valid_loader, all_classes=self.classes, title="val-set")

        sg_trainer_utils.log_uncaught_exceptions(logger)

        if not self.load_checkpoint or self.load_weights_only:
            self.start_epoch = 0
            self._reset_best_metric()
            load_opt_params = False

        if self.lr_mode is not None:
            lr_scheduler_callback = create_lr_scheduler_callback(
                lr_mode=self.lr_mode,
                train_loader=self.train_loader,
                net=self.net,
                training_params=self.training_params,
                update_param_groups=self.update_param_groups,
                optimizer=self.optimizer,
            )
            self.phase_callbacks.append(lr_scheduler_callback)

            if isinstance(lr_scheduler_callback, LRSchedulerCallback):
                self._torch_lr_scheduler = lr_scheduler_callback.scheduler
                if self.load_checkpoint:
                    self._torch_lr_scheduler.load_state_dict(self.checkpoint["torch_scheduler_state_dict"])

        if self.training_params.clip_grad_norm is not None and self.training_params.clip_grad_norm <= 0:
            raise TypeError("Params", "Invalid clip_grad_norm")

        if self.load_checkpoint and load_opt_params:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])

        self.pre_prediction_callback = CallbacksFactory().get(self.training_params.pre_prediction_callback)

        self.training_params.mixed_precision = self._initialize_mixed_precision(self.training_params.mixed_precision)

        self.ckpt_best_name = self.training_params.ckpt_best_name

        self.max_train_batches = self.training_params.max_train_batches
        self.max_valid_batches = self.training_params.max_valid_batches

        if self.training_params.max_train_batches is not None:
            if self.training_params.max_train_batches > len(self.train_loader):
                logger.warning("max_train_batches is greater than len(self.train_loader) and will have no effect.")
                self.max_train_batches = len(self.train_loader)
            elif self.training_params.max_train_batches <= 0:
                raise ValueError("max_train_batches must be positive.")

        if self.training_params.max_valid_batches is not None:
            if self.training_params.max_valid_batches > len(self.valid_loader):
                logger.warning("max_valid_batches is greater than len(self.valid_loader) and will have no effect.")
                self.max_valid_batches = len(self.valid_loader)
            elif self.training_params.max_valid_batches <= 0:
                raise ValueError("max_valid_batches must be positive.")

        self._first_backward = True

        context = PhaseContext(
            optimizer=self.optimizer,
            net=self.net,
            experiment_name=self.experiment_name,
            ckpt_dir=self.checkpoints_dir_path,
            criterion=self.criterion,
            lr_warmup_epochs=self.training_params.lr_warmup_epochs,
            sg_logger=self.sg_logger,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            training_params=self.training_params,
            ddp_silent_mode=self.ddp_silent_mode,
            checkpoint_params=self.checkpoint_params,
            architecture=self.architecture,
            arch_params=self.arch_params,
            metric_to_watch=self.metric_to_watch,
            device=device_config.device,
            ema_model=self.ema_model,
            valid_metrics=self.valid_metrics,
        )
        self.phase_callback_handler.on_training_start(context)

        model = unwrap_model(context.net)
        if (
            context.training_params.phase_callbacks is not None
            and "SlidingWindowValidationCallback" in context.training_params.phase_callbacks
            and (not hasattr(model, "enable_sliding_window_validation") or not hasattr(model, "disable_sliding_window_validation"))
        ):
            raise ValueError(
                "You can use sliding window validation callback, but your model does not support sliding window "
                "inference. Please either remove the callback or use the model that supports sliding inference: "
                "Segformer"
            )

        if isinstance(model, SupportsInputShapeCheck):
            first_train_batch = next(iter(self.train_loader))
            inputs, _, _ = sg_trainer_utils.unpack_batch_items(first_train_batch)
            model.validate_input_shape(inputs.size())

            first_valid_batch = next(iter(self.valid_loader))
            inputs, _, _ = sg_trainer_utils.unpack_batch_items(first_valid_batch)
            model.validate_input_shape(inputs.size())

        log_main_training_params(
            multi_gpu=device_config.multi_gpu,
            num_gpus=get_world_size(),
            batch_size=batch_size,
            batch_accumulate=self.batch_accumulate,
            train_dataset_length=len(self.train_loader.dataset),
            train_dataloader_len=len(self.train_loader),
            max_train_batches=self.max_train_batches,
            model=unwrap_model(self.net),
            param_groups=self.optimizer.param_groups,
        )

        self._maybe_set_preprocessing_params_for_model_from_dataset()

        try:

            if not silent_mode:
                logger.info(f"Started training for {self.max_epochs - self.start_epoch} epochs ({self.start_epoch}/" f"{self.max_epochs - 1})\n")
            for epoch in range(self.start_epoch, self.max_epochs):

                timer.start()
                if broadcast_from_master(context.stop_training):
                    logger.info("Request to stop training has been received, stopping training")
                    break


                context.update_context(epoch=epoch)
                self.phase_callback_handler.on_train_loader_start(context)


                if (
                    device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL
                    and hasattr(self.train_loader, "sampler")
                    and hasattr(self.train_loader.sampler, "set_epoch")
                ):
                    self.train_loader.sampler.set_epoch(epoch)

                train_metrics_tuple = self._train_epoch(context=context, silent_mode=silent_mode)


                train_metrics_dict = get_metrics_dict(train_metrics_tuple, self.train_metrics, self.loss_logging_items_names)

                context.update_context(metrics_dict=train_metrics_dict)
                self.phase_callback_handler.on_train_loader_end(context)


                if self.precise_bn:
                    compute_precise_bn_stats(
                        model=self.net, loader=self.train_loader, precise_bn_batch_size=self.precise_bn_batch_size, num_gpus=get_world_size()
                    )
                    if self.ema:
                        compute_precise_bn_stats(
                            model=self.ema_model.ema,
                            loader=self.train_loader,
                            precise_bn_batch_size=self.precise_bn_batch_size,
                            num_gpus=get_world_size(),
                        )


                if self.ema:
                    self.ema_model.update_attr(self.net)
                    keep_model = self.net
                    self.net = self.ema_model.ema

                train_inf_time = timer.stop()
                self._write_scalars_to_logger(metrics=train_metrics_dict, epoch=epoch, inference_time=train_inf_time, tag="Train")


                valid_metrics_dict = {}
                should_run_validation = self._should_run_validation_for_epoch(epoch)

                if should_run_validation:
                    self.phase_callback_handler.on_validation_loader_start(context)
                    timer.start()
                    valid_metrics_dict = self._validate_epoch(context=context, silent_mode=silent_mode)
                    val_inf_time = timer.stop()

                    self.valid_monitored_values = sg_trainer_utils.update_monitored_values_dict(
                        monitored_values_dict=self.valid_monitored_values,
                        new_values_dict=valid_metrics_dict,
                    )
                    context.update_context(metrics_dict=valid_metrics_dict)
                    self.phase_callback_handler.on_validation_loader_end(context)

                    self._write_scalars_to_logger(metrics=valid_metrics_dict, epoch=epoch, inference_time=val_inf_time, tag="Valid")

                test_metrics_dict = {}
                if len(self.test_loaders) and (epoch + 1) % self.run_test_freq == 0:
                    self.phase_callback_handler.on_test_loader_start(context)
                    test_inf_time = 0.0
                    for dataset_name, dataloader in self.test_loaders.items():
                        timer.start()
                        dataset_metrics_dict = self._test_epoch(data_loader=dataloader, context=context, silent_mode=silent_mode, dataset_name=dataset_name)
                        test_inf_time += timer.stop()
                        dataset_metrics_dict_with_name = {
                            f"{dataset_name}:{metric_name}": metric_value for metric_name, metric_value in dataset_metrics_dict.items()
                        }
                        self.test_monitored_values = sg_trainer_utils.update_monitored_values_dict(
                            monitored_values_dict=self.test_monitored_values,
                            new_values_dict=dataset_metrics_dict_with_name,
                        )

                        test_metrics_dict.update(**dataset_metrics_dict_with_name)
                    context.update_context(metrics_dict=test_metrics_dict)
                    self.phase_callback_handler.on_test_loader_end(context)

                    self._write_scalars_to_logger(metrics=test_metrics_dict, epoch=epoch, inference_time=test_inf_time, tag="Test")

                if self.ema:
                    self.net = keep_model

                if not self.ddp_silent_mode:
                    self.sg_logger.add_scalars(tag_scalar_dict=self._epoch_start_logging_values, global_step=epoch)


                    if should_run_validation and self.training_params.save_model:
                        self._save_checkpoint(
                            optimizer=self.optimizer,
                            epoch=1 + epoch,
                            train_metrics_dict=train_metrics_dict,
                            validation_results_dict=valid_metrics_dict,
                            context=context,
                        )
                    self.sg_logger.upload()

                if not silent_mode:
                    sg_trainer_utils.display_epoch_summary(
                        epoch=context.epoch,
                        n_digits=4,
                        monitored_values_dict={
                            "Train": self.train_monitored_values,
                            "Validation": self.valid_monitored_values,
                            "Test": self.test_monitored_values,
                        },
                    )


            self.phase_callback_handler.on_average_best_models_validation_start(context)


            if self.training_params.average_best_models:
                self._validate_final_average_model(context=context, checkpoint_dir_path=self.checkpoints_dir_path, cleanup_snapshots_pkl_file=True)

            self.phase_callback_handler.on_average_best_models_validation_end(context)

        except KeyboardInterrupt:
            context.update_context(stop_training=True)
            logger.info(
                "\n[MODEL TRAINING EXECUTION HAS BEEN INTERRUPTED]... Please wait until SOFT-TERMINATION process "
                "finishes and saves all of the Model Checkpoints and log files before terminating..."
            )
            logger.info("For HARD Termination - Stop the process again")

        finally:
            if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:

                if torch.distributed.is_initialized() and self.training_params.kill_ddp_pgroup_on_end:
                    torch.distributed.destroy_process_group()

            self.phase_callback_handler.on_training_end(context)

            if not self.ddp_silent_mode:
                self.sg_logger.close()

    def _maybe_set_preprocessing_params_for_model_from_dataset(self):
        processing_params = self._get_preprocessing_from_valid_loader()
        if processing_params is not None:
            unwrap_model(self.net).set_dataset_processing_params(**processing_params)

    def _get_preprocessing_from_valid_loader(self) -> Optional[dict]:
        valid_loader = self.valid_loader

        if isinstance(unwrap_model(self.net), HasPredict) and isinstance(valid_loader.dataset, HasPreprocessingParams):
            try:
                return valid_loader.dataset.get_dataset_preprocessing_params()
            except Exception as e:
                logger.warning(
                    f"Could not set preprocessing pipeline from the validation dataset:\n {e}.\n Before calling"
                    "predict make sure to call set_dataset_processing_params."
                )

    def _reset_best_metric(self):
        self.best_metric = -1 * np.inf if self.greater_metric_to_watch_is_better else np.inf

    def _reset_metrics(self):
        for metric in ("train_metrics", "valid_metrics", "test_metrics"):
            if hasattr(self, metric) and getattr(self, metric) is not None:
                getattr(self, metric).reset()

    @resolve_param("train_metrics_list", ListFactory(MetricsFactory()))
    def _set_train_metrics(self, train_metrics_list):
        self.train_metrics = MetricCollection(train_metrics_list)

        for metric_name, metric in self.train_metrics.items():
            if hasattr(metric, "greater_component_is_better"):
                self.greater_train_metrics_is_better.update(metric.greater_component_is_better)
            elif hasattr(metric, "greater_is_better"):
                self.greater_train_metrics_is_better[metric_name] = metric.greater_is_better
            else:
                self.greater_train_metrics_is_better[metric_name] = None

    @resolve_param("valid_metrics_list", ListFactory(MetricsFactory()))
    def _set_valid_metrics(self, valid_metrics_list):
        self.valid_metrics = MetricCollection(valid_metrics_list)

        for metric_name, metric in self.valid_metrics.items():
            if hasattr(metric, "greater_component_is_better"):
                self.greater_valid_metrics_is_better.update(metric.greater_component_is_better)
            elif hasattr(metric, "greater_is_better"):
                self.greater_valid_metrics_is_better[metric_name] = metric.greater_is_better
            else:
                self.greater_valid_metrics_is_better[metric_name] = None

    @resolve_param("test_metrics_list", ListFactory(MetricsFactory()))
    def _set_test_metrics(self, test_metrics_list):
        self.test_metrics = MetricCollection(test_metrics_list)

    def _initialize_mixed_precision(self, mixed_precision_enabled: bool):
        if mixed_precision_enabled and not device_config.is_cuda:
            warnings.warn("Mixed precision training is not supported on CPU. Disabling mixed precision. (i.e. `mixed_precision=False`)")
            mixed_precision_enabled = False


        self.scaler = GradScaler(enabled=mixed_precision_enabled)

        if mixed_precision_enabled:
            if device_config.multi_gpu == MultiGPUMode.DATA_PARALLEL:

                def hook(module, _):
                    module.forward = MultiGPUModeAutocastWrapper(module.forward)

                unwrap_model(self.net).register_forward_pre_hook(hook=hook)

            if self.load_checkpoint:
                scaler_state_dict = core_utils.get_param(self.checkpoint, "scaler_state_dict")
                if scaler_state_dict is None:
                    logger.warning("Mixed Precision - scaler state_dict not found in loaded model. This may case issues " "with loss scaling")
                else:
                    self.scaler.load_state_dict(scaler_state_dict)
        return mixed_precision_enabled

    def _validate_final_average_model(self, context: PhaseContext, checkpoint_dir_path: str, cleanup_snapshots_pkl_file=False):

        logger.info("RUNNING ADDITIONAL TEST ON THE AVERAGED MODEL...")

        keep_state_dict = deepcopy(self.net.state_dict())
        average_model_ckpt_path = os.path.join(checkpoint_dir_path, self.average_model_checkpoint_filename)
        local_rank = get_local_rank()


        with wait_for_the_master(local_rank):
            average_model_sd = read_ckpt_state_dict(average_model_ckpt_path)["net"]

        unwrap_model(self.net).load_state_dict(average_model_sd)

        context.update_context(epoch=self.max_epochs)
        averaged_model_results_dict = self._validate_epoch(context=context)
        self.valid_monitored_values = sg_trainer_utils.update_monitored_values_dict(
            monitored_values_dict=self.valid_monitored_values,
            new_values_dict=averaged_model_results_dict,
        )
        self.net.load_state_dict(keep_state_dict)

        if not self.ddp_silent_mode:
            write_struct = ""
            for name, value in averaged_model_results_dict.items():
                write_struct += "%s: %.3f  \n  " % (name, value)
                self.sg_logger.add_scalar(name, value, global_step=self.max_epochs)

            self.sg_logger.add_text("Averaged_Model_Performance", write_struct, self.max_epochs)
            if cleanup_snapshots_pkl_file:
                self.model_weight_averaging.cleanup()

    @property
    def get_arch_params(self):
        return self.arch_params.to_dict()

    @property
    def get_structure(self):
        return unwrap_model(self.net).structure

    @property
    def get_architecture(self):
        return self.architecture

    def set_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name

    def _re_build_model(self, arch_params={}):

        if "num_classes" not in arch_params.keys():
            if self.dataset_interface is None:
                raise Exception("Error", "Number of classes not defined in arch params and dataset is not defined")
            else:
                arch_params["num_classes"] = len(self.classes)

        self.arch_params = core_utils.HpmStruct(**arch_params)
        self.classes = self.arch_params.num_classes
        self.net = self._instantiate_net(self.architecture, self.arch_params, self.checkpoint_params)

        if hasattr(self.net, "structure"):
            self.architecture = self.net.structure

        self.net.to(device_config.device)

        if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
            logger.warning("Warning: distributed training is not supported in re_build_model()")
        if device_config.multi_gpu == MultiGPUMode.DATA_PARALLEL:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(device_config.num_gpus)))

    @property
    def get_module(self):
        return self.net

    def set_module(self, module):
        self.net = module

    def _switch_device(self, new_device):
        device_config.device = new_device
        self.net.to(device_config.device)


    def _load_checkpoint_to_model(self):
        self.checkpoint = {}
        strict_load = core_utils.get_param(self.training_params, "resume_strict_load", StrictLoad.ON)
        ckpt_name = core_utils.get_param(self.training_params, "ckpt_name", "ckpt_latest.pth")

        resume = core_utils.get_param(self.training_params, "resume", False)
        run_id = core_utils.get_param(self.training_params, "run_id", None)
        resume_path = core_utils.get_param(self.training_params, "resume_path")
        resume_from_remote_sg_logger = core_utils.get_param(self.training_params, "resume_from_remote_sg_logger", False)
        self.load_checkpoint = resume or (run_id is not None) or (resume_path is not None) or resume_from_remote_sg_logger

        if run_id is None:
            if resume and not (resume_from_remote_sg_logger or resume_path):

                run_id = get_latest_run_id(checkpoints_root_dir=self.ckpt_root_dir, experiment_name=self.experiment_name)
                logger.info("Resuming training from latest run.")
            else:
                run_id = generate_run_id()
                logger.info(f"Starting a new run with `run_id={run_id}`")
        else:
            validate_run_id(ckpt_root_dir=self.ckpt_root_dir, experiment_name=self.experiment_name, run_id=run_id)
            logger.info(f"Resuming training from `run_id={run_id}`")

        self.checkpoints_dir_path = get_checkpoints_dir_path(ckpt_root_dir=self.ckpt_root_dir, experiment_name=self.experiment_name, run_id=run_id)
        logger.info(f"Checkpoints directory: {self.checkpoints_dir_path}")

        with wait_for_the_master(get_local_rank()):
            if resume_from_remote_sg_logger and not self.ddp_silent_mode:
                self.sg_logger.download_remote_ckpt(ckpt_name=ckpt_name)

        if self.load_checkpoint or resume_path:
            checkpoint_path = resume_path if resume_path else os.path.join(self.checkpoints_dir_path, ckpt_name)


            self.checkpoint = load_checkpoint_to_model(
                ckpt_local_path=checkpoint_path,
                load_backbone=self.load_backbone,
                net=self.net,
                strict=strict_load.value if isinstance(strict_load, StrictLoad) else strict_load,
                load_weights_only=self.load_weights_only,
                load_ema_as_net=False,
            )

            if "ema_net" in self.checkpoint.keys():
                logger.warning(
                    "[WARNING] Main network has been loaded from checkpoint but EMA network exists as "
                    "well. It "
                    " will only be loaded during validation when training with ema=True. "
                )


        self.best_metric = self.checkpoint["acc"] if "acc" in self.checkpoint.keys() else -1
        self.start_epoch = self.checkpoint["epoch"] if "epoch" in self.checkpoint.keys() else 0

    def _prep_for_test(
        self, test_loader: torch.utils.data.DataLoader = None, loss=None, test_metrics_list=None, loss_logging_items_names=None, test_phase_callbacks=None
    ):

        self.net.eval()


        self.test_loader = test_loader or self.test_loader
        self.criterion = loss or self.criterion
        self.loss_logging_items_names = loss_logging_items_names or self.loss_logging_items_names
        self.phase_callbacks = test_phase_callbacks or self.phase_callbacks

        if self.phase_callbacks is None:
            self.phase_callbacks = []

        if test_metrics_list:
            self._set_test_metrics(test_metrics_list)
            self._add_metrics_update_callback(Phase.TEST_BATCH_END)
            self.phase_callback_handler = CallbackHandler(self.phase_callbacks)


        if self.criterion is None:
            self.loss_logging_items_names = []

        if self.test_metrics is None:
            raise ValueError(
                "Metrics are required to perform test. Pass them through test_metrics_list arg when "
                "calling test or through training_params when calling train(...)"
            )
        if test_loader is None:
            raise ValueError("Test dataloader is required to perform test. Make sure to either pass it through " "test_loader arg.")


        self._reset_metrics()
        self.test_metrics.to(device_config.device)

        if self.arch_params is None:
            self._init_arch_params()
        self._net_to_device()

    def _add_metrics_update_callback(self, phase: Phase):
        """
        Adds MetricsUpdateCallback to be fired at phase

        :param phase: Phase for the metrics callback to be fired at
        """
        self.phase_callbacks.append(MetricsUpdateCallback(phase))

    def _initialize_sg_logger_objects(self, additional_configs_to_log: Dict = None):
        """Initialize object that collect, write to disk, monitor and store remotely all training outputs"""
        sg_logger = core_utils.get_param(self.training_params, "sg_logger")

        general_sg_logger_params = {
            "experiment_name": self.experiment_name,
            "storage_location": "local",
            "resumed": self.load_checkpoint,
            "training_params": self.training_params,
            "checkpoints_dir_path": self.checkpoints_dir_path,
        }

        if sg_logger is None:
            raise RuntimeError("sg_logger must be defined in training params (see default_training_params)")

        if isinstance(sg_logger, AbstractSGLogger):
            self.sg_logger = sg_logger
        elif isinstance(sg_logger, str):
            sg_logger_cls = SG_LOGGERS.get(sg_logger)
            if sg_logger_cls is None:
                raise RuntimeError(f"sg_logger={sg_logger} not registered in SuperGradients. Available {list(SG_LOGGERS.keys())}")

            sg_logger_params = core_utils.get_param(self.training_params, "sg_logger_params", {})
            if issubclass(sg_logger_cls, BaseSGLogger):
                sg_logger_params = {**sg_logger_params, **general_sg_logger_params}

            if "model_name" in get_callable_param_names(sg_logger_cls.__init__):
                if sg_logger_params.get("model_name") is None:
                    sg_logger_params["model_name"] = get_model_name(unwrap_model(self.net))

                if sg_logger_params["model_name"] is None:
                    raise ValueError(
                        f'`model_name` is required to use `training_hyperparams.sg_logger="{sg_logger}"`.\n'
                        'Please set `training_hyperparams.sg_logger_params.model_name="<your-model-name>"`.\n'
                        "Note that specifying `model_name` is not required when the model was loaded using `models.get(...)`."
                    )

            self.sg_logger = sg_logger_cls(**sg_logger_params)
        else:
            raise RuntimeError("sg_logger can be either an sg_logger name (str) or an instance of AbstractSGLogger")

        if not isinstance(self.sg_logger, BaseSGLogger):
            logger.warning(
                "WARNING! Using a user-defined sg_logger: files will not be automatically written to disk!\n"
                "Please make sure the provided sg_logger writes to disk or compose your sg_logger to BaseSGLogger"
            )

        hyper_param_config = self._get_hyper_param_config()
        if additional_configs_to_log is not None:
            hyper_param_config["additional_configs_to_log"] = additional_configs_to_log
        self.sg_logger.add_config("hyper_params", hyper_param_config)
        self.sg_logger.flush()

    def _get_hyper_param_config(self):

        additional_log_items = {
            "initial_LR": self.training_params.initial_lr,
            "num_devices": get_world_size(),
            "multi_gpu": str(device_config.multi_gpu),
            "device_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        if self.training_params.log_installed_packages:
            additional_log_items["installed_packages"] = get_installed_packages()

        dataset_params = {
            "train_dataset_params": self.train_loader.dataset.dataset_params if hasattr(self.train_loader.dataset, "dataset_params") else None,
            "train_dataloader_params": self.train_loader.dataloader_params if hasattr(self.train_loader, "dataloader_params") else None,
            "valid_dataset_params": self.valid_loader.dataset.dataset_params if hasattr(self.valid_loader.dataset, "dataset_params") else None,
            "valid_dataloader_params": self.valid_loader.dataloader_params if hasattr(self.valid_loader, "dataloader_params") else None,
        }
        hyper_param_config = {
            "checkpoint_params": self.checkpoint_params.__dict__,
            "training_hyperparams": self.training_params.__dict__,
            "dataset_params": dataset_params,
            "additional_log_items": additional_log_items,
        }
        return hyper_param_config

    def _write_scalars_to_logger(self, metrics: dict, epoch: int, inference_time: float, tag: str) -> None:

        if not self.ddp_silent_mode:
            info_dict = {f"{tag} Inference Time": inference_time, **{f"{tag}_{k}": v for k, v in metrics.items()}}

            self.sg_logger.add_scalars(tag_scalar_dict=info_dict, global_step=epoch)

    def _get_epoch_start_logging_values(self) -> dict:

        lrs = [self.optimizer.param_groups[i]["lr"] for i in range(len(self.optimizer.param_groups))]
        lr_titles = (
            ["LR/" + self.optimizer.param_groups[i].get("name", str(i)) for i in range(len(self.optimizer.param_groups))]
            if len(self.optimizer.param_groups) > 1
            else ["LR"]
        )
        lr_dict = {lr_titles[i]: lrs[i] for i in range(len(lrs))}
        return lr_dict

    def test(
        self,
        model: nn.Module = None,
        ort_sess = None,
        test_loader: torch.utils.data.DataLoader = None,
        loss: torch.nn.modules.loss._Loss = None,
        silent_mode: bool = False,
        test_metrics_list=None,
        loss_logging_items_names=None,
        metrics_progress_verbose=False,
        test_phase_callbacks=None,
        use_ema_net=True,
        calib_path="",
        calib_num=128
    ) -> Dict[str, float]:

        self.net = model or self.net

        self.ort_sess = ort_sess

        if use_ema_net and self.ema_model is not None:
            keep_model = self.net
            self.net = self.ema_model.ema

        self._prep_for_test(
            test_loader=test_loader,
            loss=loss,
            test_metrics_list=test_metrics_list,
            loss_logging_items_names=loss_logging_items_names,
            test_phase_callbacks=test_phase_callbacks,
        )

        context = PhaseContext(
            criterion=self.criterion,
            device=self.device,
            sg_logger=self.sg_logger,
        )
        if test_metrics_list:
            context.update_context(test_metrics=self.test_metrics)
        if test_phase_callbacks:
            context.update_context(net=self.net)
            context.update_context(test_loader=test_loader)

        self.phase_callback_handler.on_test_loader_start(context)
        test_results = self.evaluate(
            data_loader=test_loader,
            metrics=self.test_metrics,
            evaluation_type=EvaluationType.TEST,
            silent_mode=silent_mode,
            metrics_progress_verbose=metrics_progress_verbose,
            calib_path=calib_path,
            calib_num=calib_num
        )
        self.phase_callback_handler.on_test_loader_end(context)

        if use_ema_net and self.ema_model is not None:
            self.net = keep_model

        self._first_backward = True

        return test_results

    def _validate_epoch(self, context: PhaseContext, silent_mode: bool = False) -> Dict[str, float]:

        self.net.eval()
        self._reset_metrics()
        self.valid_metrics.to(device_config.device)
        return self.evaluate(
            data_loader=self.valid_loader,
            metrics=self.valid_metrics,
            evaluation_type=EvaluationType.VALIDATION,
            epoch=context.epoch,
            silent_mode=silent_mode,
            max_batches=self.max_valid_batches,
        )

    def _test_epoch(self, data_loader: DataLoader, context: PhaseContext, silent_mode: bool = False, dataset_name: str = "") -> Dict[str, float]:

        self.net.eval()
        self._reset_metrics()
        self.test_metrics.to(device_config.device)
        return self.evaluate(
            data_loader=data_loader,
            metrics=self.test_metrics,
            evaluation_type=EvaluationType.TEST,
            epoch=context.epoch,
            silent_mode=silent_mode,
            dataset_name=dataset_name,
        )

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        metrics: MetricCollection,
        evaluation_type: EvaluationType,
        epoch: int = None,
        silent_mode: bool = False,
        metrics_progress_verbose: bool = False,
        dataset_name: str = "",
        max_batches: Optional[int] = None,
        calib_path: str = "",
        calib_num: int = 128
    ) -> Dict[str, float]:

        loss_avg_meter = core_utils.utils.AverageMeter()

        lr_warmup_epochs = self.training_params.lr_warmup_epochs if self.training_params else None
        context = PhaseContext(
            net=self.net,
            epoch=epoch,
            metrics_compute_fn=metrics,
            loss_avg_meter=loss_avg_meter,
            criterion=self.criterion,
            device=device_config.device,
            lr_warmup_epochs=lr_warmup_epochs,
            sg_logger=self.sg_logger,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            loss_logging_items_names=self.loss_logging_items_names,
        )

        expected_iterations = len(data_loader) if max_batches is None else max_batches

        with tqdm(
            data_loader, total=expected_iterations, bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True, disable=silent_mode
        ) as progress_bar_data_loader:
            if not silent_mode:

                pbar_start_msg = "Validating" if evaluation_type == EvaluationType.VALIDATION else "Testing"
                if dataset_name:
                    pbar_start_msg += f' dataset="{dataset_name}:"'
                if epoch:
                    pbar_start_msg += f" epoch {epoch}"
                progress_bar_data_loader.set_description(pbar_start_msg)
            with torch.no_grad():
                for batch_idx, batch_items in enumerate(progress_bar_data_loader):
                    if evaluation_type == EvaluationType.VALIDATION and expected_iterations <= batch_idx:
                        break

                    batch_items = core_utils.tensor_container_to_device(batch_items, device_config.device, non_blocking=True)
                    inputs, targets, additional_batch_items = sg_trainer_utils.unpack_batch_items(batch_items)


                    context.update_context(
                        batch_idx=batch_idx, inputs=inputs, target=targets, additional_batch_items=additional_batch_items, **additional_batch_items
                    )
                    if evaluation_type == EvaluationType.VALIDATION:
                        self.phase_callback_handler.on_validation_batch_start(context)
                    else:
                        self.phase_callback_handler.on_test_batch_start(context)

                    if calib_path != "":
                        inputs_one = inputs.cpu().numpy()
                        save_path = f"{calib_path}/sample_{batch_idx}.npy"
                        np.save(save_path, inputs_one)

                    if calib_path != "" and batch_idx < calib_num:
                        continue

                    if calib_path != "" and batch_idx >= calib_num:
                        break

                    if self.ort_sess is not None:
                        item_device = inputs.device
                        input_item = {"input.1": inputs.cpu().numpy()}
                        output = self.ort_sess.run(None, input_item)
                        output1 = torch.tensor(np.array(output[0])).to(item_device)
                        output2 = torch.tensor(np.array(output[1])).to(item_device)
                        output = ((output1, output2))
                    else:
                        output = self.net(inputs)

                    context.update_context(preds=output)

                    if self.criterion is not None:

                        loss_tuple = self._get_losses(output, targets)[1].cpu()
                        context.update_context(loss_log_items=loss_tuple)


                    if evaluation_type == EvaluationType.VALIDATION:
                        self.phase_callback_handler.on_validation_batch_end(context)
                    else:
                        self.phase_callback_handler.on_test_batch_end(context)


                    if metrics_progress_verbose and not silent_mode:

                        logging_values = get_logging_values(loss_avg_meter, metrics, self.criterion)
                        pbar_message_dict = get_train_loop_description_dict(logging_values, metrics, self.loss_logging_items_names)

                        progress_bar_data_loader.set_postfix(**pbar_message_dict)

            logging_values = get_logging_values(loss_avg_meter, metrics, self.criterion)

            if not metrics_progress_verbose:

                pbar_message_dict = get_train_loop_description_dict(logging_values, metrics, self.loss_logging_items_names)

                progress_bar_data_loader.set_postfix(**pbar_message_dict)


            if device_config.multi_gpu == MultiGPUMode.DISTRIBUTED_DATA_PARALLEL:
                logging_values = reduce_results_tuple_for_ddp(logging_values, next(self.net.parameters()).device)

        return get_train_loop_description_dict(logging_values, metrics, self.loss_logging_items_names)

    def _instantiate_net(
        self, architecture: Union[torch.nn.Module, SgModule.__class__, str], arch_params: dict, checkpoint_params: dict, *args, **kwargs
    ) -> tuple:

        pretrained_weights = core_utils.get_param(checkpoint_params, "pretrained_weights", default_val=None)

        if pretrained_weights is not None:
            num_classes_new_head = arch_params.num_classes
            arch_params.num_classes = PRETRAINED_NUM_CLASSES[pretrained_weights]

        if isinstance(architecture, str):
            architecture_cls = ARCHITECTURES[architecture]
            net = architecture_cls(arch_params=arch_params)
        elif isinstance(architecture, SgModule.__class__):
            net = architecture(arch_params)
        else:
            net = architecture

        if pretrained_weights:
            load_pretrained_weights(net, architecture, pretrained_weights)
            if num_classes_new_head != arch_params.num_classes:
                net.replace_head(new_num_classes=num_classes_new_head)
                arch_params.num_classes = num_classes_new_head

        return net

    def _instantiate_ema_model(self, ema_params: Mapping[str, Any]) -> ModelEMA:

        logger.info(f"Using EMA with params {ema_params}")
        return ModelEMA.from_params(self.net, **ema_params)

    @property
    def get_net(self):

        return self.net

    def set_net(self, net: torch.nn.Module):
        self.net = net

    def set_ckpt_best_name(self, ckpt_best_name):
        self.ckpt_best_name = ckpt_best_name

    def set_ema(self, val: bool):
        self.ema = val

    def _init_loss_logging_names(self, loss_logging_items):
        criterion_name = self.criterion.__class__.__name__
        component_names = None
        if hasattr(self.criterion, "component_names"):
            component_names = self.criterion.component_names
        elif len(loss_logging_items) > 1:
            component_names = ["loss_" + str(i) for i in range(len(loss_logging_items))]

        if component_names is not None:
            self.loss_logging_items_names = [criterion_name + "/" + component_name for component_name in component_names]
            if self.metric_to_watch in component_names:
                self.metric_to_watch = criterion_name + "/" + self.metric_to_watch
        else:
            self.loss_logging_items_names = [criterion_name]

    @classmethod
    def quantize_from_config(cls, cfg: Union[DictConfig, dict]) -> Tuple[nn.Module, Tuple]:

        if _imported_pytorch_quantization_failure is not None:
            raise _imported_pytorch_quantization_failure


        cfg = hydra.utils.instantiate(cfg)


        cfg = cls._trigger_cfg_modifying_callbacks(cfg)

        quantization_params = get_param(cfg, "quantization_params")

        if quantization_params is None:
            logger.warning("Your recipe does not include quantization_params. Using default quantization params.")
            quantization_params = load_recipe("quantization_params/default_quantization_params").quantization_params
            cfg.quantization_params = quantization_params

        if get_param(cfg.checkpoint_params, "checkpoint_path") is None and get_param(cfg.checkpoint_params, "pretrained_weights") is None:
            raise ValueError("Starting checkpoint / pretrained weights are a must for QAT finetuning.")

        num_gpus = core_utils.get_param(cfg, "num_gpus")
        multi_gpu = core_utils.get_param(cfg, "multi_gpu")
        device = core_utils.get_param(cfg, "device")
        if num_gpus != 1:
            raise NotImplementedError(
                f"Recipe requests multi_gpu={cfg.multi_gpu} and num_gpus={cfg.num_gpus}. QAT is proven to work correctly only with multi_gpu=OFF and num_gpus=1"
            )

        setup_device(device=device, multi_gpu=multi_gpu, num_gpus=num_gpus)


        train_dataloader = dataloaders.get(
            name=get_param(cfg, "train_dataloader"),
            dataset_params=copy.deepcopy(cfg.dataset_params.train_dataset_params),
            dataloader_params=copy.deepcopy(cfg.dataset_params.train_dataloader_params),
        )

        val_dataloader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=copy.deepcopy(cfg.dataset_params.val_dataset_params),
            dataloader_params=copy.deepcopy(cfg.dataset_params.val_dataloader_params),
        )

        if "calib_dataloader" in cfg:
            calib_dataloader_name = get_param(cfg, "calib_dataloader")
            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.calib_dataloader_params)
            calib_dataset_params = copy.deepcopy(cfg.dataset_params.calib_dataset_params)
        else:
            calib_dataloader_name = get_param(cfg, "train_dataloader")
            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.train_dataloader_params)
            calib_dataset_params = copy.deepcopy(cfg.dataset_params.train_dataset_params)


            calib_dataloader_params.shuffle = cfg.quantization_params.calib_params.num_calib_batches is not None

            calib_dataset_params.transforms = cfg.dataset_params.val_dataset_params.transforms

        calib_dataloader = dataloaders.get(
            name=calib_dataloader_name,
            dataset_params=calib_dataset_params,
            dataloader_params=calib_dataloader_params,
        )

        model = models.get(
            model_name=cfg.arch_params.get("model_name", None) or cfg.architecture,
            num_classes=cfg.get("num_classes", None) or cfg.arch_params.num_classes,
            arch_params=cfg.arch_params,
            strict_load=cfg.checkpoint_params.strict_load,
            pretrained_weights=cfg.checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.checkpoint_params.checkpoint_path,
            load_backbone=False,
            checkpoint_num_classes=get_param(cfg.checkpoint_params, "checkpoint_num_classes"),
            num_input_channels=get_param(cfg.arch_params, "num_input_channels"),
        )

        recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}
        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=get_param(cfg, "ckpt_root_dir"))

        if quantization_params.ptq_only:
            res = trainer.ptq(
                calib_loader=calib_dataloader,
                model=model,
                quantization_params=quantization_params,
                valid_loader=val_dataloader,
                valid_metrics_list=cfg.training_hyperparams.valid_metrics_list,
            )
        else:
            res = trainer.qat(
                model=model,
                quantization_params=quantization_params,
                calib_loader=calib_dataloader,
                valid_loader=val_dataloader,
                train_loader=train_dataloader,
                training_params=cfg.training_hyperparams,
                additional_qat_configs_to_log=recipe_logged_cfg,
            )

        return model, res

    def qat(
        self,
        calib_loader: DataLoader,
        model: torch.nn.Module,
        valid_loader: DataLoader,
        train_loader: DataLoader,
        training_params: Mapping = None,
        quantization_params: Mapping = None,
        additional_qat_configs_to_log: Dict = None,
        valid_metrics_list: List[Metric] = None,
    ):

        if quantization_params is None:
            quantization_params = load_recipe("quantization_params/default_quantization_params").quantization_params
            logger.info(f"Using default quantization params: {quantization_params}")
        valid_metrics_list = valid_metrics_list or get_param(training_params, "valid_metrics_list")

        _ = self.ptq(
            calib_loader=calib_loader,
            model=model,
            quantization_params=quantization_params,
            valid_loader=valid_loader,
            valid_metrics_list=valid_metrics_list,
            deepcopy_model_for_export=True,
        )

        model.train()
        torch.cuda.empty_cache()

        res = self.train(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            training_params=training_params,
            additional_configs_to_log=additional_qat_configs_to_log,
        )

        input_shape = next(iter(valid_loader))[0].shape
        os.makedirs(self.checkpoints_dir_path, exist_ok=True)
        qdq_onnx_path = os.path.join(self.checkpoints_dir_path, f"{self.experiment_name}_{'x'.join((str(x) for x in input_shape))}_qat.onnx")

        export_quantized_module_to_onnx(
            model=model.cpu(),
            onnx_filename=qdq_onnx_path,
            input_shape=input_shape,
            input_size=input_shape,
            train=False,
        )
        logger.info(f"Exported QAT ONNX to {qdq_onnx_path}")
        return res

    def ptq(
        self,
        calib_loader: DataLoader,
        model: nn.Module,
        valid_loader: DataLoader,
        valid_metrics_list: List[torchmetrics.Metric],
        quantization_params: Dict = None,
        deepcopy_model_for_export: bool = False,
    ):

        logger.debug("Performing post-training quantization (PTQ)...")
        logger.debug(f"Experiment name {self.experiment_name}")

        run_id = core_utils.get_param(self.training_params, "run_id", None)
        logger.debug(f"Experiment run id {run_id}")

        self.checkpoints_dir_path = get_checkpoints_dir_path(ckpt_root_dir=self.ckpt_root_dir, experiment_name=self.experiment_name, run_id=run_id)
        logger.debug(f"Checkpoints directory {self.checkpoints_dir_path}")

        os.makedirs(self.checkpoints_dir_path, exist_ok=True)

        from super_gradients.training.utils.quantization.fix_pytorch_quantization_modules import patch_pytorch_quantization_modules_if_needed

        patch_pytorch_quantization_modules_if_needed()

        if quantization_params is None:
            quantization_params = load_recipe("quantization_params/default_quantization_params").quantization_params
            logger.info(f"Using default quantization params: {quantization_params}")

        model = unwrap_model(model)
        model = model.to(device_config.device).eval()

        selective_quantizer_params = get_param(quantization_params, "selective_quantizer_params")
        calib_params = get_param(quantization_params, "calib_params")

        fuse_repvgg_blocks_residual_branches(model)
        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights=get_param(selective_quantizer_params, "calibrator_w"),
            default_quant_modules_calibrator_inputs=get_param(selective_quantizer_params, "calibrator_i"),
            default_per_channel_quant_weights=get_param(selective_quantizer_params, "per_channel"),
            default_learn_amax=get_param(selective_quantizer_params, "learn_amax"),
            verbose=get_param(calib_params, "verbose"),
        )
        q_util.register_skip_quantization(layer_names=get_param(selective_quantizer_params, "skip_modules"))
        q_util.quantize_module(model)

        logger.info("Calibrating model...")
        calibrator = QuantizationCalibrator(
            verbose=get_param(calib_params, "verbose"),
            torch_hist=True,
        )
        calibrator.calibrate_model(
            model,
            method=get_param(calib_params, "histogram_calib_method"),
            calib_data_loader=calib_loader,
            num_calib_batches=get_param(calib_params, "num_calib_batches") or len(calib_loader),
            percentile=get_param(calib_params, "percentile", 99.99),
        )
        calibrator.reset_calibrators(model)

        logger.info("Validating PTQ model...")
        valid_metrics_dict = self.test(model=model, test_loader=valid_loader, test_metrics_list=valid_metrics_list)
        results = ["PTQ Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in valid_metrics_dict.items()]
        logger.info("\n".join(results))

        input_shape = next(iter(valid_loader))[0].shape
        input_shape_with_batch_size_one = tuple([1] + list(input_shape[1:]))
        qdq_onnx_path = os.path.join(
            self.checkpoints_dir_path, f"{self.experiment_name}_{'x'.join((str(x) for x in input_shape_with_batch_size_one))}_ptq.onnx"
        )
        logger.debug(f"Output ONNX file path {qdq_onnx_path}")

        if isinstance(model, ExportableObjectDetectionModel):
            model: ExportableObjectDetectionModel = typing.cast(ExportableObjectDetectionModel, model)
            export_result = model.export(
                output=qdq_onnx_path,
                quantization_mode=ExportQuantizationMode.INT8,
                input_image_shape=(input_shape_with_batch_size_one[2], input_shape_with_batch_size_one[3]),
                preprocessing=False,
                postprocessing=True,
            )
            logger.info(repr(export_result))
        else:

            export_quantized_module_to_onnx(
                model=model.cpu(),
                onnx_filename=qdq_onnx_path,
                input_shape=input_shape_with_batch_size_one,
                input_size=input_shape_with_batch_size_one,
                train=False,
                deepcopy_model=deepcopy_model_for_export,
            )

        return valid_metrics_dict
