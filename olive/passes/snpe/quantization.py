# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from olive.cache import get_local_path_from_root
from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import SNPEModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.platform_sdk.qualcomm.snpe.tools.dev import quantize_dlc
from olive.platform_sdk.qualcomm.utils.data_loader import FileListCommonDataLoader, FileListDataLoader
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, LocalFile
from olive.strategy.search_parameter import Boolean


class SNPEQuantization(Pass):
    """Quantize SNPE model.

    Uses snpe-dlc-quantize tool from the SNPE SDK.
    """

    _requires_user_script = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "data_dir": PassConfigParam(
                type_=OLIVE_RESOURCE_ANNOTATIONS,
                required=False,
                category=ParamCategory.DATA,
                description="Path to the data directory. Required is data_config is None.",
            ),
            "dataloader_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                category=ParamCategory.OBJECT,
                description=(
                    "Function or function name to create dataloader for quantization. Function should take data"
                    " directory as an argument and return a FileListDataLoader or torch.data.DataLoader-like"
                    " object. Required if data_config is None."
                ),
            ),
            "dataloader_func_kwargs": PassConfigParam(
                type_=Dict[str, Any],
                description="Keyword arguments for dataloader_func.",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                description="Data config for quantization, required if dataloader_func is None",
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
                description="Pack HTP information in quantized DLC, which is not available in Windows.",
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
                    " for more additional arguments. The value is a string that will be passed as is to the tool."
                    " e.g.: --bias_bitwidth 16 --overwrite_cache_records"
                ),
            ),
        }

    def _run_for_config(
        self, model: SNPEModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> SNPEModelHandler:
        if Path(output_model_path).suffix != ".dlc":
            output_model_path += ".dlc"

        assert config["dataloader_func"] or config["data_config"], "dataloader_func or data_config is required."

        if config["dataloader_func"]:
            data_dir = get_local_path_from_root(data_root, config["data_dir"])
            dataloader = self._user_module_loader.call_object(
                config["dataloader_func"], data_dir, **(config["dataloader_func_kwargs"] or {})
            )
        elif config["data_config"]:
            data_config = validate_config(config["data_config"], DataConfig)
            dataloader = data_config.to_data_container().create_dataloader(data_root)

        # convert dataloader to FileListDataLoader if it is not already
        if not isinstance(dataloader, FileListDataLoader):
            dataloader = FileListCommonDataLoader(dataloader, model.io_config)

        quantize_dlc(model.model_path, dataloader.get_input_list(), config, output_model_path)
        return SNPEModelHandler(model_path=LocalFile({"path": output_model_path}), **model.io_config)
