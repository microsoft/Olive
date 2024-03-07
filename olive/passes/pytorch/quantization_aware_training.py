# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Union

from olive.cache import get_local_path_from_root
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import PyTorchModelHandler
from olive.passes import Pass
from olive.passes.olive_pass import ParamCategory, PassConfigParam


class QuantizationAwareTraining(Pass):
    """Run quantization aware training on PyTorch model."""

    _requires_user_script = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        import pytorch_lightning
        from packaging import version

        if version.parse(pytorch_lightning.__version__) >= version.parse("1.9.0"):
            from pytorch_lightning.loggers import Logger
        else:
            from pytorch_lightning.loggers import LightningLoggerBase as Logger
        return {
            "train_data_dir": PassConfigParam(
                type_=str, description="Directory of training data.", category=ParamCategory.DATA
            ),
            "val_data_dir": PassConfigParam(
                type_=str, description="Directory of validation data.", category=ParamCategory.DATA
            ),
            "train_dataloader_func": PassConfigParam(
                type_=Union[Callable, str],
                category=ParamCategory.OBJECT,
                description=(
                    "Dataloader function to load training data from given train_data_dir with given train_batch_size."
                ),
            ),
            "training_loop_func": PassConfigParam(
                type_=Union[Callable, str],
                ategory=ParamCategory.OBJECT,
                description="Customized training loop function.",
            ),
            "ptl_module": PassConfigParam(
                type_=Union[Callable, str],
                category=ParamCategory.OBJECT,
                description=(
                    "LightningModule for PyTorch Lightning trainer. It is a way of encapsulating all the logic "
                    "related to the training, validation, and testing of a PyTorch model. Please refer to "
                    "https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html for more details."
                ),
            ),
            "ptl_data_module": PassConfigParam(
                type_=Union[Callable, str],
                category=ParamCategory.OBJECT,
                description=(
                    "LightningDataModule for PyTorch Lightning trainer. It is a way of encapsulating all the "
                    "data-related logic for training, validation, and testing of a PyTorch model. Please refer to "
                    "https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html for more details."
                ),
            ),
            "train_batch_size": PassConfigParam(type_=int, description="Batch size for training."),
            "num_epochs": PassConfigParam(type_=int, description="Maximum number of epochs for training."),
            "num_steps": PassConfigParam(
                type_=int, default_value=-1, description="Maximum number of steps for training."
            ),
            "do_validate": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether perform one evaluation epoch over the validation set after training.",
            ),
            "modules_to_fuse": PassConfigParam(
                type_=List[List[str]], default_value=None, description="List of list of module names to fuse."
            ),
            "qconfig_func": PassConfigParam(
                type_=Union[Callable, str],
                default_value=None,
                category=ParamCategory.OBJECT,
                description=(
                    "Customized function to create a QConfig for QAT. Please refer to "
                    "https://pytorch.org/docs/stable/generated/torch.ao.quantization.qconfig.QConfig.html for details."
                ),
            ),
            "logger": PassConfigParam(
                type_=Union[Logger, Iterable[Logger], Callable, bool],
                required=False,
                default_value=False,
                category=ParamCategory.OBJECT,
                description="Logger for training.",
            ),
            "gpus": PassConfigParam(type_=int, description="Number of GPUs to use."),
            "seed": PassConfigParam(type_=int, default_value=None, description="Random seed for training."),
            "checkpoint_path": PassConfigParam(type_=str, default_value=None, description="Path to save checkpoints."),
        }

    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> PyTorchModelHandler:
        from olive.passes.pytorch.qat_utils import QatTrainer

        qat_trainer_config = self._config_class(**config)
        if Path(output_model_path).suffix != ".pt":
            output_model_path += ".pt"

        if config["train_dataloader_func"]:
            qat_trainer_config.train_dataloader_func = self._user_module_loader.load_object(
                config["train_dataloader_func"]
            )
        qat_trainer_config.train_data_dir = get_local_path_from_root(data_root, qat_trainer_config.train_data_dir)

        if config["training_loop_func"]:
            qat_trainer_config.training_loop_func = self._user_module_loader.load_object(config["training_loop_func"])
        qat_trainer_config.val_data_dir = get_local_path_from_root(data_root, qat_trainer_config.val_data_dir)
        if config["ptl_module"]:
            qat_trainer_config.ptl_module = self._user_module_loader.load_object(config["ptl_module"])
        if config["ptl_data_module"]:
            qat_trainer_config.ptl_data_module = self._user_module_loader.load_object(config["ptl_data_module"])
        if config["qconfig_func"]:
            qat_trainer_config.qconfig_func = self._user_module_loader.load_object(config["qconfig_func"])

        qat_trainer = QatTrainer(model, qat_trainer_config, output_model_path)
        return qat_trainer.execute_local()
