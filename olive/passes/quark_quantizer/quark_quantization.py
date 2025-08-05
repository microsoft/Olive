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

import torch

from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

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
        }

    def _run_for_config(self, model: HfModelHandler, config: BasePassConfig, output_model_path: str) -> HfModelHandler:
        logger.info("[INFO] Running QuarkQuantization with config: %s", config)

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
