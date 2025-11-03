#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import logging
import platform
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

import onnx
import torch
from packaging import version

from olive.common.config_utils import validate_config
from olive.common.utils import exclude_keys
from olive.data.config import DataConfig
from olive.model import HfModelHandler, ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.search.search_parameter import Categorical

logger = logging.getLogger(__name__)


class QuarkQuantization(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec=None):
        return {
            "quant_scheme": PassConfigParam(
                type_=str, default_value="w_uint4_per_group_asym", description="Quantization scheme to use."
            ),
            "quant_algo": PassConfigParam(type_=str, default_value="awq", description="Quantization algorithm."),
            "dataset": PassConfigParam(
                type_=str, default_value="pileval_for_awq_benchmark", description="Calibration dataset to use."
            ),
            "data_type": PassConfigParam(type_=str, default_value="float32", description="Data type for model."),
            "num_calib_data": PassConfigParam(
                type_=int, default_value=128, description="Number of calibration samples."
            ),
            "model_export": PassConfigParam(
                type_=list, default_value=["hf_format"], description="Model export format."
            ),
            "exclude_layers": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of layers to exclude. Set to [] to exclude nothing explicitly.",
            ),
            "quant_config": PassConfigParam(
                type_=dict, default_value=None, description="Embedded quant configuration dictionary"
            ),
            "quant_mode": PassConfigParam(
                type_=str,
                default_value="static",
                search_defaults=Categorical(["dynamic", "static"]),
                description="Onnx Quantization mode. 'dynamic' for dynamic quantization, 'static' for static quantization. Default is 'static'",
            ),
            "quant_format": PassConfigParam(
                type_=str,
                default_value="QDQ",
                search_defaults=Categorical(["QOperator", "QDQ"]),
                description="Onnx Quantization format. 'QOperator' for quantizing models using QOperators, 'QDQ' for using Q/DQ. Default is 'QDQ'",
            ),
            "data_config": PassConfigParam(
                type_=Optional[Union[DataConfig, dict]],
                default_value=None,
                description="Data config for calibration.",
            ),
            "global_config": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Global quantization configuration applied to all layers unless overridden.",
            ),
            "specific_layer_config": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Dictionary mapping specific layer names to their quantization configuration. Default is None.",
            ),
            "layer_type_config": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Dictionary mapping layer types (e.g., Conv, Gemm) to quantization configurations. Default is None.",
            ),
            "exclude": PassConfigParam(
                type_=dict,
                default_value=None,
                description="List of nodes or subgraphs excluded from quantization. Default is None.",
            ),
            "algo_config": PassConfigParam(
                type_=list,
                default_value=None,
                description="Algorithm configuration, can be a list of algorithm configurations. Default is None.",
            ),
            "extra_options": PassConfigParam(
                type_=dict, default_value=None, description="Extra options for quantization. Default is {}."
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: Union[HfModelHandler, ONNXModelHandler], config: BasePassConfig, output_model_path: str
    ) -> Union[HfModelHandler, ONNXModelHandler]:
        if isinstance(model, ONNXModelHandler):
            logger.info("[INFO] Running QuarkQuantization using Quark-ONNX API with config: %s", config)
            return self._run_quark_onnx(model, config, output_model_path)
        else:
            logger.info("[INFO] Running QuarkQuantization using Quark-Torch API with config: %s", config)
            return self._run_quark_torch(model, config, output_model_path)

    def _run_quark_onnx(
        self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        from quark import __version__ as QuarkVersion

        if version.parse(QuarkVersion) < version.parse("0.10.0"):
            raise ValueError("Quark onnx Quantization is only supported for amd-quark>=0.10.0")

        from olive.passes.quark_quantizer.onnx.quantize_quark import run_quark_quantization

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # to be safe, run the quantizer with use_external_data_format set to `True` and
        # `model_output` to a temporary directory
        # reload the model and save to output_model_path using the external data config
        new_tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")  # pylint: disable=R1732
        tmp_model_path = str(Path(new_tmp_dir.name) / Path(output_model_path).name)

        data_reader = None
        if config.data_config:
            data_config = validate_config(config.data_config, DataConfig)
            data_reader = data_config.to_data_container().create_calibration_dataloader()

        run_config = config.dict()
        if config.extra_options is None:
            run_config["extra_options"] = {}
        if data_reader is None:
            run_config["extra_options"]["UseRandomData"] = True

        to_delete = [
            "data_config",
            "quant_preprocess",
        ]
        to_delete += list(get_external_data_config().keys())
        run_config = exclude_keys(run_config, to_delete)

        args = Namespace(
            model_input=model.model_path,
            model_output=tmp_model_path,
            calibration_data_reader=data_reader,
            **run_config,
        )

        run_quark_quantization(args)
        logger.info("[INFO] Quark quantized model saved to: %s", tmp_model_path)

        # load the model
        onnx_model = onnx.load(tmp_model_path)
        # the model is loaded into memory, so it's safe to delete previously exported files
        new_tmp_dir.cleanup()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    def _run_quark_torch(self, model: HfModelHandler, config: BasePassConfig, output_model_path: str) -> HfModelHandler:
        from olive.passes.quark_quantizer.torch.language_modeling.llm_ptq.quantize_quark import run_quark_quantization

        output_dir = Path(output_model_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        if platform.system().lower() == "linux" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        quant_algo_config_file_path = None
        if config.quant_config:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp_file:
                json.dump(config.quant_config, tmp_file)
                quant_algo_config_file_path = tmp_file.name
                logger.info("[INFO] Written quant_config to temporary file: %s", quant_algo_config_file_path)

        args = Namespace(
            model_dir=str(model.model_path),
            output_dir=str(output_model_path),
            quant_scheme=config.quant_scheme,
            quant_algo=config.quant_algo,
            dataset=config.dataset,
            data_type=config.data_type,
            num_calib_data=config.num_calib_data,
            model_export=config.model_export,
            exclude_layers=config.exclude_layers,
            device=device,
            quant_algo_config_file_path=quant_algo_config_file_path,
            # Other args
            multi_gpu=False,
            model_attn_implementation="eager",
            multi_device=False,
            skip_quantization=False,
            group_size=128,
            group_size_per_layer=None,
            kv_cache_dtype=None,
            min_kv_scale=0.0,
            pre_quantization_optimization=[],
            pre_optimization_config_file_path=None,
            scale_format="e4m3",
            scale_calculation_mode="even",
            fp8_attention_quant=False,
            moe_experts_second_step_config=None,
            model_reload=False,
            import_model_dir=None,
            params_load=False,
            json_path=None,
            safetensors_path=None,
            import_file_format="quark_format",
            custom_mode="quark",
            torch_compile=False,
            pack_method="reorder",
            weight_matrix_merge=False,
            export_weight_format="real_quantized",
            params_save=False,
            save_dir="model_params",
            skip_evaluation=True,
            save_metrics_to_csv=False,
            metrics_output_dir="metrics_output_dir",
            tasks=None,
            use_ppl_eval_for_kv_cache=False,
            ppl_eval_for_kv_cache_context_size=1024,
            ppl_eval_for_kv_cache_sample_size=512,
            ppl_eval_for_kv_cache_patch_size=None,
            eval_batch_size="8",
            max_eval_batch_size=None,
            num_eval_data=-1,
            num_fewshot=None,
            apply_chat_template=False,
            use_mlperf_rouge=False,
            eval_data_dir=None,
            use_tp=False,
            seq_len=512,
            batch_size=1,
        )

        run_quark_quantization(args)
        logger.info("[INFO] Quark quantized model saved to: %s", output_model_path)
        # Cleanup
        if quant_algo_config_file_path:
            tmp_path = Path(quant_algo_config_file_path)
            if tmp_path.exists():
                tmp_path.unlink()
            logger.info("[INFO] Deleted temporary quant config file: %s", quant_algo_config_file_path)

        return HfModelHandler(str(output_model_path))
