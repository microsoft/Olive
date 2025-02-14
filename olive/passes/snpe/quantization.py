# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Type, Union

from olive.common.config_utils import validate_config
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import SNPEModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.platform_sdk.qualcomm.snpe.tools.dev import quantize_dlc
from olive.platform_sdk.qualcomm.utils.data_loader import FileListCommonDataLoader, FileListDataLoader
from olive.resource_path import LocalFile
from olive.search.search_parameter import Boolean


class SNPEQuantization(Pass):
    """Quantize SNPE model.

    Uses snpe-dlc-quantize tool from the SNPE SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "data_config": PassConfigParam(
                type_=Union[DataConfig, Dict],
                required=True,
                description="Data config for quantization",
            ),
            "use_enhanced_quantizer": PassConfigParam(
                type_=bool,
                default_value=False,
                search_defaults=Boolean(),
                description=(
                    "Use the enhanced quantizer feature when quantizing the model. Uses an algorithm to determine"
                    " optimal range instead of min and max range of data.  It can be useful for quantizing models that"
                    " have long tails in the distribution of the data being quantized."
                ),
            ),
            "enable_htp": PassConfigParam(
                type_=bool,
                default_value=False,
                search_defaults=Boolean(),
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
        self, model: SNPEModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> SNPEModelHandler:
        if Path(output_model_path).suffix != ".dlc":
            output_model_path += ".dlc"

        data_config = validate_config(config.data_config, DataConfig)
        dataloader = data_config.to_data_container().create_dataloader()

        # convert dataloader to FileListDataLoader if it is not already
        if not isinstance(dataloader, FileListDataLoader):
            dataloader = FileListCommonDataLoader(dataloader, model.io_config)

        quantize_dlc(model.model_path, dataloader.get_input_list(), config, output_model_path)
        return SNPEModelHandler(model_path=LocalFile({"path": output_model_path}), **model.io_config)
