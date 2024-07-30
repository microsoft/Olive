# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy

import pytorch_lightning
import torch
import torch.quantization.quantization_mappings as tqqm
from packaging import version
from pytorch_lightning import LightningModule, seed_everything
from torch.ao.quantization.fake_quantize import FakeQuantize, MovingAverageMinMaxObserver

from olive.common.config_utils import validate_config
from olive.constants import ModelFileFormat
from olive.data.config import DataConfig
from olive.model import PyTorchModelHandler
from olive.passes.pytorch.cluster import barrier, create_cluster, is_master_proc
from olive.passes.pytorch.pytorch_lightning_utils import create_ddp_strategy, create_trainer


class QuantizedModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


class QatTrainer:
    def __init__(self, model, config, output_model_path):
        self.model = model
        self.config = config
        self.output_model_path = output_model_path

    def execute_local(self) -> PyTorchModelHandler:
        seed_everything(self.config.seed)
        cluster_environment = create_cluster()
        run_on_gpus = cluster_environment is not None or torch.cuda.is_available()
        input_model = self.model.load_model()
        model_input_tensor = self.model.get_dummy_inputs()

        if self.config.qconfig_func:
            qconfig = self.config.qconfig_func()
        else:
            default_fake_quant = FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=False,
            )
            qconfig = torch.quantization.QConfig(
                activation=default_fake_quant, weight=torch.quantization.default_weight_fake_quant
            )

        quan_model = self.prepare_qat(input_model, qconfig)

        if self.config.training_loop_func:
            quantized_model = self.config.training_loop_func(model=quan_model)

        else:
            ptl_data_module = None
            if self.config.ptl_module:
                ptl_module = self.config.ptl_module(model=quan_model)
                if self.config.ptl_data_module:
                    ptl_data_module = self.config.ptl_data_module()
            else:
                train_data_config = validate_config(self.config.train_data_config, DataConfig)
                train_dataloader = train_data_config.to_data_container().create_dataloader()
                ptl_module = DefaultPTLModule(model=quan_model, training_dataloader=train_dataloader)

            kwargs = {}
            if run_on_gpus:
                num_gpus = self.config.gpus if self.config.gpus is not None else torch.cuda.device_count()
                ddp = create_ddp_strategy(cluster=cluster_environment, accelerator="gpu")
                kwargs["strategy"] = ddp
                kwargs["devices"] = num_gpus
                if version.parse(pytorch_lightning.__version__) >= version.parse("1.9.0"):
                    kwargs["use_distributed_sampler"] = False
                else:
                    kwargs["replace_sampler_ddp"] = False

            trainer = create_trainer(
                max_epochs=self.config.num_epochs,
                max_steps=self.config.num_steps,
                logger=self.config.logger,
                default_root_dir=self.config.checkpoint_path,
                **kwargs
            )

            trainer.fit(ptl_module, datamodule=ptl_data_module)

            if self.config.do_validate:
                trainer.validate(ptl_module, datamodule=ptl_data_module)
            quantized_model = copy.deepcopy(ptl_module.model)

        quantized_model.eval()
        quantized_model.to("cpu")
        return self.post_qat(quantized_model, model_input_tensor)

    def prepare_qat(self, model: torch.nn.Module, qconfig: torch.quantization.QConfig) -> torch.nn.Module:
        self.fuse_modules(model)
        self.replace_modules(model, qconfig)
        model.train()
        return torch.ao.quantization.prepare_qat(model, inplace=False)

    def post_qat(self, model: torch.nn.Module, model_input_tensor) -> torch.nn.Module:
        model.apply(torch.quantization.disable_observer)
        model_converted = torch.ao.quantization.convert(model.eval(), inplace=False)
        traced_model_converted = torch.jit.trace(model_converted, model_input_tensor, strict=False)
        if is_master_proc():
            torch.jit.save(traced_model_converted, self.output_model_path)
        barrier()
        # preserve the io_config, dummy_inputs_func, model_script, script_dir
        # from the original model
        original_config = self.model.to_json()["config"]
        to_keep = ["io_config", "dummy_inputs_func"]
        if isinstance(original_config["dummy_inputs_func"], str):
            to_keep += ["model_script", "script_dir"]
        config_to_keep = {k: original_config[k] for k in to_keep}
        # TODO(jambayk): Add PyTorch model type flag
        return PyTorchModelHandler(
            model_path=self.output_model_path, model_file_format=ModelFileFormat.PYTORCH_TORCH_SCRIPT, **config_to_keep
        )

    def fuse_modules(self, model: torch.nn.Module):
        if self.config.modules_to_fuse:
            for group in self.config.modules_to_fuse:
                if self._check_feasible_fuse(model, group):
                    torch.quantization.fuse_modules(model, [group], inplace=True)
            for _, child in model.named_children():
                self.fuse_modules(child)

    def replace_modules(self, module: torch.nn.Module, qconfig: torch.quantization.QConfig, prefix: str = ""):
        white_list = tqqm.get_default_qconfig_propagation_list()
        skip_list = [
            torch.nn.Embedding,
            torch.nn.LayerNorm,
            torch.nn.Dropout,
            torch.nn.Sequential,
            torch.nn.BatchNorm2d,
        ]

        for name, child in module.named_children():
            op_name = prefix + "." + name if prefix != "" else name
            if len(list(child.children())) > 0:
                self.replace_modules(child, qconfig, op_name)
            if type(child) in white_list and type(child) not in skip_list:
                new = QuantizedModule(child)
                new.qconfig = qconfig  # pylint: disable=attribute-defined-outside-init
                setattr(module, name, new)

    def _recursive_hasattr(self, obj, attribs, state=True):
        if "." in attribs:
            attrib, attribs = attribs.split(".", 1)
            if hasattr(obj, attrib):
                return self._recursive_hasattr(getattr(obj, attrib), attribs, state)
            return False
        return state and hasattr(obj, attribs)

    def _check_feasible_fuse(self, model, group):
        return all(self._recursive_hasattr(model, m) for m in group)


class DefaultPTLModule(LightningModule):
    # pylint: disable=W0221
    def __init__(self, model, training_dataloader):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.training_dataloader = training_dataloader
        self.loss_module = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        if isinstance(data, dict):
            preds = self.model(**data)
        else:
            preds = self.model(data)
        return self.loss_module(preds, labels)

    def train_dataloader(self):
        return self.training_dataloader
