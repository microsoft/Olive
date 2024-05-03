# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Union

from olive.cache import get_local_path_from_root
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import OliveModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.pass_config import ParamCategory, PassConfigParam
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS
from olive.strategy.search_parameter import Categorical

# static quantization specific config
_dataloader_config = {
    "data_dir": PassConfigParam(
        type_=OLIVE_RESOURCE_ANNOTATIONS,
        category=ParamCategory.DATA,
        description="""
            Path to the directory containing the dataset.
            For local data, it is required if quant_mode is 'static' and dataloader_func is provided.
        """,
    ),
    "batch_size": PassConfigParam(
        type_=int,
        default_value=1,
        description="""
            Batch size for calibration, only used if dataloader_func is provided.
        """,
    ),
    "calib_size": PassConfigParam(
        type_=int,
        default_value=64,
        description="""
            Calibration data size, only used if dataloader_func is provided.
        """,
    ),
    "dataloader_func": PassConfigParam(
        type_=Union[Callable, str],
        category=ParamCategory.OBJECT,
        description="""
            Function/function name to generate dataloader for calibration,
            required if quant_mode is 'static' and data_config is None.
        """,
    ),
    "dataloader_func_kwargs": PassConfigParam(
        type_=Dict[str, Any],
        description="Keyword arguments for dataloader_func.",
    ),
}


class NVModelOptQuantization(Pass):

    # set this to True if the pass has parameters that are functions or objects and the user script is required
    # to import the module containing the function or object
    _requires_user_script: bool = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "precision": PassConfigParam(
                type_=str,
                default_value="int4",
                searchable_values=Categorical(["fp8", "int8", "int4"]),
                description="""
                    NVModelOpt Quantization mode.
                """,
            ),
            "algorithm": PassConfigParam(
                type_=str,
                default_value="RTN",
                searchable_values=Categorical(["RTN", "AWQ"]),
                description="""
                    Algorithm of weight only quantization. Support 'RTN' and 'AWQ'.
                """,
            ),
        }

        config.update(deepcopy(_dataloader_config))
        return config

    def _run_for_config(
        self, model: OliveModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> OliveModelHandler:
        try:
            from modelopt.onnx.quantization.int4 import quantize_int4  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Please install `olive-ai[nvmo]` or `nvidia-modelopt` to use INT4 AWQ quantization!"
            ) from exc

        data_dir = get_local_path_from_root(data_root, config["data_dir"])
        calib_dataloader = self._user_module_loader.call_object(
            config["dataloader_func"],
            data_dir,
            config["batch_size"],
            config["calib_size"],
            model_path=model.model_path,
            **(config["dataloader_func_kwargs"] or {}),
        )

        if config["precision"] != "int4" or config["algorithm"] not in ["RTN", "AWQ"]:
            raise ValueError("Only INT4 quantization with RTN and AWQ algorithm is supported.")

        quantize_mode = "int4_awq_clip" if config["algorithm"] == "AWQ" else "int4_rtn_dq"
        q_model = quantize_int4(quantize_mode, model.load_model(), calib_dataloader)

        # save the model to the output path and return the model
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        return model_proto_to_olive_model(q_model, output_model_path, config)
