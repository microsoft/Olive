# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import torch

from olive.common.config_utils import validate_config
from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.hf_utils import OliveHfQuantizationConfig, replace_matching_submodules
from olive.common.quant.linear import QuantLinear
from olive.common.quant.utils import WeightQuantizer
from olive.common.utils import tensor_data_to_device
from olive.constants import PrecisionBits
from olive.data.config import DataConfig
from olive.data.template import huggingface_data_config_template
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.train_utils import (
    load_hf_base_model,
)

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler


logger = logging.getLogger(__name__)

# ruff: noqa: N806


class Gptq(Pass):
    """GPTQ quantization."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS4,
                description="quantization bits. Default value is 4",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Block size for quantization. Default value is 128.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Symmetric quantization. Default value is False.",
            ),
            "damp_percent": PassConfigParam(
                type_=float,
                default_value=0.01,
                description="Damping factor for quantization. Default value is 0.01.",
            ),
            "desc_act": PassConfigParam(
                type_=bool,
                default_value=None,
                description=(
                    "Whether to use act-order (also called desc-act) scheme. True is only supported when group_size is"
                    " -1. Default is None, which is equivalent to True for group_size -1 and False for other group"
                    " sizes."
                ),
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value=None,
                description=(
                    "Data config for quantization. If not provided, wikitest train data will be used for HfModels."
                    " Required for PyTorch models."
                ),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.group_size <= 0 and config.group_size != -1:
            logger.info("group_size must be -1 or greater than 0")
            return False

        if config.desc_act is True and config.group_size != -1:
            logger.info("desc_act can only be True when group_size is -1.")
            return False

        return True

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        """Run GPTQ quantization on the model.

        Args:
            model: The HuggingFace model to quantize.
            config: Configuration object containing quantization parameters.
            output_model_path: Path where the quantized model will be saved.

        Returns:
            HfModelHandler for the quantized model.

        """
        from tqdm.auto import tqdm

        wrapper = ModelWrapper.from_model(load_hf_base_model(model, torch_dtype="auto"))
        wrapper.model.eval()
        original_use_cache = wrapper.model.config.use_cache
        wrapper.model.config.use_cache = False

        quant_config = self.get_quant_config(model, config)

        self.prepare_model(wrapper, quant_config)

        # get the inputs for the first layer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hidden_states, layer_args, layer_kwargs = self.get_init_layer_inputs(model, wrapper, config, device)

        for layer in tqdm(wrapper.get_layers(return_name=False), desc="Quantizing layers"):
            quantizable_modules = [module for module in layer.modules() if hasattr(module, "quant_info")]

            # collect calibration data
            handles = [module.register_forward_hook(self.accumulate_hessian) for module in quantizable_modules]
            self.run_layer(layer, hidden_states, layer_args, layer_kwargs)
            for handle in handles:
                handle.remove()

            # process each quantizable module
            for module in quantizable_modules:
                self.process_module(module, percdamp=config.damp_percent, actorder=config.desc_act)

            # run the layer again to get the quantized outputs
            hidden_states = self.run_layer(
                layer,
                hidden_states,
                layer_args,
                layer_kwargs,
                return_output=True,
            )

        # finalize the quantization
        self.finalize(wrapper, quant_config, device)
        wrapper.model.config.use_cache = original_use_cache

        # save the quantized model
        wrapper.model.save_pretrained(output_model_path)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)

    def get_quant_config(self, model: HfModelHandler, config: type[BasePassConfig]) -> OliveHfQuantizationConfig:
        """Get quantization configuration with mixed precision support.

        Args:
            model: The HuggingFace model to get configuration for.
            config: Configuration object containing quantization parameters.

        Returns:
            OliveHfQuantizationConfig object with quantization settings.

        """
        quant_config = {
            "bits": config.bits,
            "symmetric": config.sym,
            "group_size": config.group_size,
        }
        if mp_info := (model.model_attributes or {}).get("mixed_precision_info"):
            for k, v in quant_config.items():
                if mp_info["default"].get(k) is not None and v != mp_info["default"][k]:
                    logger.debug("Overriding %s with mixed precision info: %s", k, mp_info["default"][k])
                    quant_config[k] = mp_info["default"][k]
            quant_config["overrides"] = mp_info.get("overrides")
        return OliveHfQuantizationConfig(**quant_config)

    def prepare_model(self, wrapper: ModelWrapper, quant_config: OliveHfQuantizationConfig) -> None:
        """Prepare the model for quantization by adding quant_info to linear layers.

        Args:
            wrapper: ModelWrapper containing the model to prepare.
            quant_config: Quantization configuration to use.

        """
        # TODO(jambayk): make lm head quantization configurable
        lm_head_name = wrapper.get_lm_head()[1]

        def should_quantize(module: torch.nn.Module, name: str) -> bool:
            return isinstance(module, torch.nn.Linear) and name != lm_head_name

        def add_quant_info(module: torch.nn.Module, name: str) -> torch.nn.Module:
            # TODO(jambayk): validate that the module and config are compatible
            module.quant_info = QuantInfo(quantizer=WeightQuantizer(**quant_config.get_qlinear_init_args(name)))
            return module

        replace_matching_submodules(
            wrapper.model,
            should_quantize,
            add_quant_info,
            description="Preparing model for quantization",
        )

    @torch.no_grad()
    def get_init_layer_inputs(
        self, model: HfModelHandler, wrapper: ModelWrapper, config: type[BasePassConfig], device: str
    ) -> tuple[list[torch.Tensor], list[tuple], list[dict]]:
        """Get initial layer inputs for quantization calibration.

        Args:
            model: The HuggingFace model.
            wrapper: ModelWrapper containing the model.
            config: Configuration object containing data settings.
            device: Device to run calibration on.

        Returns:
            Tuple containing hidden states, layer args, and layer kwargs.

        """
        hidden_states, layer_args, layer_kwargs = [], [], []

        pre_layer_modules = list(wrapper.get_embeds(return_name=False))
        if rotary_embed := wrapper.get_rotary_embed(return_name=False):
            pre_layer_modules.append(rotary_embed)
        for module in pre_layer_modules:
            module.to(device)

        def store_input_hook(_, args: tuple, kwargs: dict) -> None:
            # assume first argument is the hidden state
            hidden_states.append(args[0])
            layer_args.append(args[1:])
            layer_kwargs.append(kwargs)
            raise ValueError

        first_layer = wrapper.get_layers(return_name=False)[0]
        hook = first_layer.register_forward_pre_hook(store_input_hook, with_kwargs=True)

        for data in self.get_dataset(model, config):
            try:
                wrapper.model(**tensor_data_to_device(data, device))
            except ValueError:
                # we raised ValueError to stop the forward pass
                pass

        hook.remove()
        for module in pre_layer_modules:
            module.to("cpu")

        return hidden_states, layer_args, layer_kwargs

    @torch.no_grad()
    def run_layer(
        self,
        layer: torch.nn.Module,
        hidden_states: list[torch.Tensor],
        layer_args: list[tuple],
        layer_kwargs: list[dict],
        return_output: bool = False,
    ) -> list[torch.Tensor] | None:
        """Run a layer with the given inputs.

        Args:
            layer: The model layer to run.
            hidden_states: List of hidden state tensors.
            layer_args: List of additional positional arguments for each input.
            layer_kwargs: List of keyword arguments for each input.
            return_output: Whether to return the layer outputs.

        Returns:
            List of output tensors if return_output is True, otherwise None.

        """
        outputs = []
        layer.to(hidden_states[0].device)

        for i, hs in enumerate(hidden_states):
            # TODO(jambayk): support non true-sequential if needed
            layer_output = layer(
                hs,
                *layer_args[i],
                **layer_kwargs[i],
            )[0]
            if return_output:
                outputs.append(layer_output)

        layer.to("cpu")
        return outputs or None

    @staticmethod
    def accumulate_hessian(module: torch.nn.Module, inp: tuple, _: Any) -> None:
        """Accumulate Hessian matrix for GPTQ quantization.

        Args:
            module: The linear module to accumulate Hessian for.
            inp: Input tensors to the module.
            _: Unused output parameter.

        """
        if module.quant_info.data is None:
            module.quant_info.data = {
                "H": torch.zeros((module.in_features, module.in_features), device=inp[0].device),
                "N": 0,
            }

        batch_size = inp[0].shape[0]
        inp = inp[0].reshape(-1, module.in_features).t()

        module.quant_info.data["H"] *= module.quant_info.data["N"] / (module.quant_info.data["N"] + batch_size)
        module.quant_info.data["N"] += batch_size
        inp = math.sqrt(2 / module.quant_info.data["N"]) * inp.float()
        module.quant_info.data["H"] += inp.matmul(inp.t())

    @staticmethod
    def process_module(
        module: torch.nn.Module, blocksize: int = 128, percdamp: float = 0.01, actorder: bool | None = False
    ) -> None:
        """Process a module for GPTQ quantization using the accumulated Hessian.

        Args:
            module: The linear module to quantize.
            blocksize: Block size for processing weights.
            percdamp: Damping factor for numerical stability.
            actorder: Whether to use act-order quantization scheme.

        """
        if module.quant_info.data is None:
            raise ValueError(f"Module {module} does not have quant_info.data initialized!")

        if actorder is None:
            actorder = module.quant_info.quantizer.group_size == -1
        elif actorder is True:
            assert module.quant_info.quantizer.group_size == -1, (
                "actorder can only be True when group_size is -1, but got group_size="
                f"{module.quant_info.quantizer.group_size}"
            )

        H = module.quant_info.data["H"]
        W = module.weight.data.clone().float().to(H.device)
        num_cols = H.shape[0]

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(num_cols, device=H.device)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)  # pylint: disable=not-callable
        del H
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)  # pylint: disable=not-callable

        all_scales = []
        all_zp = []
        now_idx = 1
        # create a per-channel quantizer
        quantizer = WeightQuantizer(
            bits=module.quant_info.quantizer.bits, symmetric=module.quant_info.quantizer.symmetric, group_size=-1
        )
        if module.quant_info.quantizer.group_size == -1:
            # this can be before or after actorder permutation since there's only one group
            active_scale, active_zp = quantizer.find_qparams(W)
        else:
            active_scale, active_zp = None, None

        for i1 in range(0, num_cols, blocksize):
            i2 = min(i1 + blocksize, num_cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if module.quant_info.quantizer.group_size != -1:
                    if (i1 + i) % module.quant_info.quantizer.group_size == 0:
                        active_scale, active_zp = quantizer.find_qparams(
                            W[:, (i1 + i) : (i1 + i + module.quant_info.quantizer.group_size)]
                        )

                    if ((i1 + i) // module.quant_info.quantizer.group_size) - now_idx == -1:
                        all_scales.append(active_scale)
                        all_zp.append(active_zp)
                        now_idx += 1

                q = quantizer.fake_quantize(w.unsqueeze(1), active_scale, active_zp).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        if not all_scales:
            all_scales.append(active_scale)
            all_zp.append(active_zp)

        module.weight.data = Q.to(module.weight.data.device).to(module.weight.data.dtype)
        module.quant_info.scales = torch.cat(all_scales, dim=1).to("cpu")
        module.quant_info.zero_points = torch.cat(all_zp, dim=1).to("cpu")

        module.quant_info.data = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def finalize(self, wrapper: ModelWrapper, quant_config: OliveHfQuantizationConfig, device: str) -> None:
        """Finalize quantization by replacing linear layers with quantized versions.

        Args:
            wrapper: ModelWrapper containing the model to finalize.
            quant_config: Quantization configuration to use.
            device: Device to perform quantization on.

        """

        def should_quantize(module: torch.nn.Module, _: str) -> bool:
            return hasattr(module, "quant_info")

        def quantize_and_pack(module: torch.nn.Module, _: str) -> QuantLinear:
            module.to(device)
            return QuantLinear.from_linear(
                module.to(device),
                bits=module.quant_info.quantizer.bits,
                symmetric=module.quant_info.quantizer.symmetric,
                group_size=module.quant_info.quantizer.group_size,
                scales=module.quant_info.scales,
                zero_points=module.quant_info.zero_points,
            ).to("cpu")  # move the original module to CPU

        replace_matching_submodules(
            wrapper.model,
            should_quantize,
            quantize_and_pack,
            description="Quantizing and packing linear layers",
        )

        wrapper.model.quantization_method = quant_config.quant_method
        wrapper.model.config.quantization_config = quant_config

    def get_dataset(self, model: HfModelHandler, config: type[BasePassConfig]) -> list[dict[str, Any]]:
        """Get the dataset for quantization calibration.

        Args:
            model: The HuggingFace model to get dataset for.
            config: Configuration object containing data settings.

        Returns:
            List of tokenized data dictionaries for calibration.

        Raises:
            ValueError: If the dataset format is invalid.

        """
        data_config = config.data_config or self.get_calibration_data_config(
            model.model_name_or_path, trust_remote_code=model.get_load_kwargs().get("trust_remote_code", None)
        )
        data_config = validate_config(data_config, DataConfig)
        dataloader = data_config.to_data_container().create_dataloader()
        # each batch consists of (input_data, labels)
        dataset = [data[0] for data in dataloader]

        if (
            not dataset
            or not isinstance(dataset, list)
            or not isinstance(dataset[0], dict)
            or ("input_ids" not in dataset[0] or "attention_mask" not in dataset[0])
        ):
            raise ValueError(
                "Provided dataset is invalid. The returned datasets is a list of tokenized data "
                "(e.g. [{ 'input_ids': [[ 1, 100, 15, ... ]],'attention_mask': [[ 1, 1, 1, ... ]]},...])"
            )

        return dataset

    @staticmethod
    def get_calibration_data_config(model_name_or_path: str, trust_remote_code: bool | None = None) -> DataConfig:
        """Get default calibration data configuration for GPTQ quantization.

        Args:
            model_name_or_path: Name or path of the model.
            trust_remote_code: Whether to trust remote code when loading data.

        Returns:
            DataConfig object for calibration data.

        """
        return huggingface_data_config_template(
            model_name=model_name_or_path,
            task="text-generation",
            load_dataset_config={
                "data_name": "wikitext",
                "subset": "wikitext-2-raw-v1",
                # only require 128 samples for calibration
                "split": "train[:1000]",
                "trust_remote_code": trust_remote_code,
            },
            pre_process_data_config={
                # should we randomize the data?
                "add_special_tokens": False,
                "max_seq_len": 2048,
                "max_samples": 128,
                "trust_remote_code": trust_remote_code,
            },
        )


@dataclass
class QuantInfo:
    """Class to hold quantization information for GPTQ.

    This class stores all the necessary information for quantizing a layer,
    including the quantizer, computed scales and zero points, and calibration data.

    Attributes:
        quantizer: The weight quantizer used for quantization.
        scales: Computed scales for quantization. Set after processing.
        zero_points: Computed zero points for quantization. Set after processing.
        data: Calibration data including Hessian matrix and sample count.
              Format: {"H": torch.Tensor, "N": int} or None.

    """

    quantizer: WeightQuantizer
    scales: torch.Tensor | None = None
    zero_points: torch.Tensor | None = None
    data: dict | None = None
