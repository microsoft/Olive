# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy

import torch
import torch.quantization.quantization_mappings as tqqm
from pytorch_lightning import LightningModule, seed_everything
from torch.ao.quantization.fake_quantize import FakeQuantize, MovingAverageMinMaxObserver

from olive.constants import ModelFileFormat
from olive.model import ModelStorageKind, PyTorchModel
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
        x = self.dequant(x)
        return x


class QatTrainer:
    def __init__(self, model, config, output_model_path):
        self.model = model
        self.config = config
        self.output_model_path = output_model_path

    def execute_local(self) -> PyTorchModel:
        seed_everything(self.config.seed)
        cluster_environment = create_cluster()
        run_on_gpus = cluster_environment is not None or torch.cuda.is_available()
        input_model = self.model.load_model()
        model_input_tensor = self.create_dummy_input()

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
                train_dataloader_func = self.config.train_dataloader_func(
                    self.config.train_data_dir, self.config.train_batch_size
                )
                ptl_module = DefaultPTLModule(model=quan_model, training_dataloader=train_dataloader_func)

            kwargs = {}
            if run_on_gpus:
                num_gpus = self.config.gpus if self.config.gpus is not None else torch.cuda.device_count()
                ddp = create_ddp_strategy(cluster=cluster_environment, accelerator="gpu")
                kwargs["strategy"] = ddp
                kwargs["devices"] = num_gpus
                kwargs["replace_sampler_ddp"] = False

            trainer = create_trainer(
                max_epochs=self.config.num_epochs, max_steps=self.config.num_steps, logger=self.config.logger, **kwargs
            )

            trainer.fit(ptl_module, datamodule=ptl_data_module)

            if self.config.do_validate:
                trainer.validate(ptl_module, datamodule=ptl_data_module)
            quantized_model = copy.deepcopy(ptl_module.model)

        quantized_model.eval()
        quantized_model.to("cpu")
        quantized_model = self.post_qat(quantized_model, model_input_tensor)
        return quantized_model

    def prepare_qat(self, model: torch.nn.Module, qconfig: torch.quantization.QConfig) -> torch.nn.Module:
        self.fuse_modules(model)
        self.replace_modules(model, qconfig)
        model.train()
        model_prepared = torch.ao.quantization.prepare_qat(model, inplace=False)
        return model_prepared

    def post_qat(self, model: torch.nn.Module, model_input_tensor) -> torch.nn.Module:
        model.apply(torch.quantization.disable_observer)
        model_converted = torch.ao.quantization.convert(model.eval(), inplace=False)
        traced_model_converted = torch.jit.trace(model_converted, model_input_tensor, strict=False)
        if is_master_proc():
            torch.jit.save(traced_model_converted, self.output_model_path)
        barrier()
        # TODO: Add PyTorch model type flag
        qat_pytorch_model = PyTorchModel(
            model_path=self.output_model_path,
            name="pytorch_qat_model",
            model_file_format=ModelFileFormat.PYTORCH_TORCH_SCRIPT,
            model_storage_kind=ModelStorageKind.LocalFile,
        )
        return qat_pytorch_model

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
            if type(child) in white_list:
                if type(child) not in skip_list:
                    if not type(child) in skip_list:
                        new = QuantizedModule(child)
                        new.qconfig = qconfig
                        setattr(module, name, new)

    def _recursive_hasattr(self, obj, attribs, state=True):
        if "." in attribs:
            attrib, attribs = attribs.split(".", 1)
            if hasattr(obj, attrib):
                return self._recursive_hasattr(getattr(obj, attrib), attribs, state)
            return False
        return state and hasattr(obj, attribs)

    def _check_feasible_fuse(self, model, group):
        if not all(self._recursive_hasattr(model, m) for m in group):
            return False
        return True

    def create_dummy_input(self):
        str_to_type = {"float32": torch.float32, "float16": torch.float16, "int32": torch.int32, "int64": torch.int64}
        input_types = []
        if self.config.input_types is not None:
            for input_type in self.config.input_types:
                input_types.append(str_to_type[input_type])
        else:
            input_types = [str_to_type["float32"] for _ in self.config.input_shapes]

        # dummy inputs
        dummy_inputs = []
        for input_shape, input_type in zip(self.config.input_shapes, input_types):
            dummy_inputs.append(torch.zeros(input_shape, dtype=input_type))
        dummy_inputs = tuple(dummy_inputs) if len(dummy_inputs) > 1 else dummy_inputs[0]
        return dummy_inputs


class DefaultPTLModule(LightningModule):
    def __init__(self, model, training_dataloader):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.training_dataloader = training_dataloader
        self.loss_module = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, labels = batch
        if isinstance(data, dict):
            preds = self.model(**data)
        else:
            preds = self.model(data)
        loss = self.loss_module(preds, labels)
        return loss

    def train_dataloader(self):
        return self.training_dataloader
