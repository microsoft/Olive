# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import SNPEModel
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam
from olive.resource_path import LocalFile
from olive.snpe import SNPECommonDataLoader, SNPEDataLoader
from olive.snpe.tools.dev import quantize_dlc
from olive.strategy.search_parameter import Boolean


class SNPEQuantization(Pass):
    """
    Quantize SNPE model.
    Uses snpe-dlc-quantize tool from the SNPE SDK.
    """

    _requires_user_script = True
    _requires_data_config = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "data_dir": PassConfigParam(
                type_=str,
                required=False,
                is_path=True,
                description="Path to the data directory. Required is data_config is None.",
            ),
            "dataloader_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                is_object=True,
                description=(
                    "Function or function name to create dataloader for quantization. Function should take data"
                    " directory as an argument and return a olive.snpe.SNPEDataLoader or torch.data.DataLoader-like"
                    " object. Required if data_config is None."
                ),
            ),
            "use_enhanced_quantizer": PassConfigParam(
                type_=bool,
                default_value=False,
                searchable_values=Boolean(),
                description=(
                    "Use the enhanced quantizer feature when quantizing the model. Uses an algorithm to determine"
                    " optimal range instead of min and max range of data.  It can be useful for quantizing models that"
                    " have long tails in the distribution of the data being quantized."
                ),
            ),
            "enable_htp": PassConfigParam(
                type_=bool,
                default_value=False,
                searchable_values=Boolean(),
                description="Pack HTP information in quantized DLC.",
            ),
            "htp_socs": PassConfigParam(
                type_=List[str], default_value=None, description="List of SoCs to generate HTP Offline cache for."
            ),
            "extra_args": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Extra arguments to pass to snpe conversion tool. Refer to"
                    " https://developer.qualcomm.com/sites/default/files/docs/snpe/tools.html#tools_snpe-dlc-quantize"
                    " for more additional arguments. Must be a dictionary of the form: {'arg_name': 'arg_value'}."
                ),
            ),
        }

    def _run_for_config(self, model: SNPEModel, config: Dict[str, Any], output_model_path: str) -> SNPEModel:
        if Path(output_model_path).suffix != ".dlc":
            output_model_path += ".dlc"

        if config["dataloader_func"]:
            dataloader = self._user_module_loader.call_object(config["dataloader_func"], config["data_dir"])
        elif self._data_config:
            dataloader = self._data_config.to_data_container().create_dataloader()

        # convert dataloader to SNPEDataLoader if it is not already
        if not isinstance(dataloader, SNPEDataLoader):
            dataloader = SNPECommonDataLoader(dataloader, model.io_config)

        quantize_dlc(model.model_path, dataloader.get_input_list(), config, output_model_path)
        return SNPEModel(model_path=LocalFile({"path": output_model_path}), **model.io_config)
