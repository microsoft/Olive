#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

import onnx
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
    """Quark Quantization Pass for Olive.

    Routes to the appropriate backend based on input model type:
    - ONNXModelHandler  -> Quark-ONNX quantization (quark.onnx)
    - HfModelHandler    -> Quark-Torch quantization (quark.torch, Quark 0.11 API)
    """

    @classmethod
    def _default_config(cls, accelerator_spec=None):
        return {
            # ── Torch-specific configs ───────────────────────────────────
            "quant_scheme": PassConfigParam(
                type_=str,
                default_value="uint4_wo_128",
                description="[Torch] Quantization scheme: uint4_wo_128, int4_wo_128, int8, fp8, mxfp4",
            ),
            "quant_algo": PassConfigParam(
                type_=Union[str, list],
                default_value="awq",
                description="[Torch] Quantization algorithm(s): awq, gptq, smoothquant, rotation",
            ),
            "dataset": PassConfigParam(
                type_=str,
                default_value="pileval_for_awq_benchmark",
                description="[Torch] Calibration dataset",
            ),
            "data_type": PassConfigParam(
                type_=str,
                default_value="bfloat16",
                description="[Torch] Model data type: auto, float16, bfloat16, float32",
            ),
            "num_calib_data": PassConfigParam(
                type_=int,
                default_value=128,
                description="[Torch] Number of calibration samples",
            ),
            "model_export": PassConfigParam(
                type_=Union[str, list],
                default_value="hf_format",
                description="[Torch] Export format(s): hf_format, onnx, gguf",
            ),
            "exclude_layers": PassConfigParam(
                type_=list,
                default_value=None,
                description="[Torch] Layers to exclude from quantization",
            ),
            "layer_quant_scheme": PassConfigParam(
                type_=list,
                default_value=None,
                description="[Torch] Layer-specific schemes: [['lm_head', 'int8']]",
            ),
            "kv_cache_dtype": PassConfigParam(
                type_=str,
                default_value=None,
                description="[Torch] KV cache dtype: fp8 or None",
            ),
            "min_kv_scale": PassConfigParam(
                type_=float,
                default_value=0.0,
                description="[Torch] Minimum scale for KV cache quantization",
            ),
            "kv_cache_post_rope": PassConfigParam(
                type_=bool,
                default_value=False,
                description="[Torch] Quantize KV cache after RoPE",
            ),
            "attention_dtype": PassConfigParam(
                type_=str,
                default_value=None,
                description="[Torch] Attention quantization dtype: fp8 or None",
            ),
            "seq_len": PassConfigParam(
                type_=int,
                default_value=512,
                description="[Torch] Sequence length for calibration",
            ),
            "batch_size": PassConfigParam(
                type_=int,
                default_value=1,
                description="[Torch] Batch size for calibration",
            ),
            "pack_method": PassConfigParam(
                type_=str,
                default_value="reorder",
                description="[Torch] Pack method: order or reorder",
            ),
            "export_weight_format": PassConfigParam(
                type_=str,
                default_value="real_quantized",
                description="[Torch] Weight format: fake_quantized or real_quantized",
            ),
            "custom_mode": PassConfigParam(
                type_=str,
                default_value="quark",
                description="[Torch] Export mode: quark, awq, fp8",
            ),
            "trust_remote_code": PassConfigParam(
                type_=bool,
                default_value=True,
                description="[Torch] Trust remote code from HuggingFace",
            ),
            # ── ONNX-specific configs ────────────────────────────────────
            "quant_mode": PassConfigParam(
                type_=str,
                default_value="static",
                search_defaults=Categorical(["dynamic", "static"]),
                description="[ONNX] Onnx Quantization mode. 'dynamic' for dynamic quantization, 'static' for static quantization. Default is 'static'",
            ),
            "quant_format": PassConfigParam(
                type_=str,
                default_value="QDQ",
                search_defaults=Categorical(["QOperator", "QDQ"]),
                description="[ONNX] Onnx Quantization format. 'QOperator' for quantizing models using QOperators, 'QDQ' for using Q/DQ. Default is 'QDQ'",
            ),
            "data_config": PassConfigParam(
                type_=Optional[Union[DataConfig, dict]],
                default_value=None,
                description="[ONNX] Data config for calibration.",
            ),
            "global_config": PassConfigParam(
                type_=dict,
                default_value=None,
                description="[ONNX] Global quantization configuration applied to all layers unless overridden.",
            ),
            "specific_layer_config": PassConfigParam(
                type_=list,
                default_value=None,
                description="[ONNX] List of specific layer configurations. Default is None.",
            ),
            "layer_type_config": PassConfigParam(
                type_=list,
                default_value=None,
                description="[ONNX] List of layer type configurations. Default is None.",
            ),
            "exclude": PassConfigParam(
                type_=list,
                default_value=None,
                description="[ONNX] List of nodes or subgraphs excluded from quantization. Default is None.",
            ),
            "algo_config": PassConfigParam(
                type_=list,
                default_value=None,
                description="[ONNX] Algorithm configuration, can be a list of algorithm configurations. Default is None.",
            ),
            "extra_options": PassConfigParam(
                type_=dict,
                default_value=None,
                description="[ONNX] Extra options for quantization. Default is {}.",
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self,
        model: Union[HfModelHandler, ONNXModelHandler],
        config: BasePassConfig,
        output_model_path: str,
    ) -> Union[HfModelHandler, ONNXModelHandler]:
        if isinstance(model, ONNXModelHandler):
            logger.info("[INFO] Running QuarkQuantization using Quark-ONNX API")
            return self._run_quark_onnx(model, config, output_model_path)
        else:
            logger.info("[INFO] Running QuarkQuantization using Quark-Torch 0.11 API")
            return self._run_quark_torch(model, config, output_model_path)

    # ── ONNX path ───────────────────────────────────────────

    def _run_quark_onnx(
        self,
        model: ONNXModelHandler,
        config: BasePassConfig,
        output_model_path: str,
    ) -> ONNXModelHandler:
        from quark import __version__ as QuarkVersion

        if version.parse(QuarkVersion) < version.parse("0.11.0"):
            raise ValueError("Quark ONNX Quantization is only supported for amd-quark>=0.11.0")

        from olive.passes.quark_quantizer.onnx.quantize_quark import run_quark_quantization

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # Run quantizer to a temp dir, then reload and save with the external data config
        new_tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")  # pylint: disable=R1732
        tmp_model_path = str(Path(new_tmp_dir.name) / Path(output_model_path).name)

        data_reader = None
        if config.data_config:
            data_config = validate_config(config.data_config, DataConfig)
            data_reader = data_config.to_data_container().create_calibration_dataloader()

        run_config = config.model_dump()
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
        logger.info("[INFO] Quark ONNX quantized model saved to: %s", tmp_model_path)

        onnx_model = onnx.load(tmp_model_path)
        new_tmp_dir.cleanup()

        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    # ── Torch path (delegates to torch module) ───────────────────────────

    def _run_quark_torch(
        self,
        model: HfModelHandler,
        config: BasePassConfig,
        output_model_path: str,
    ) -> HfModelHandler:
        from olive.passes.quark_quantizer.torch.quark_torch_quantization import run_quark_torch_quantization

        return run_quark_torch_quantization(model, config, output_model_path)
