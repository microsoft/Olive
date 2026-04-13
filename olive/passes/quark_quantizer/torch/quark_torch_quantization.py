#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Quark 0.11 Torch quantization for LLMs.

Uses LLMTemplate + ModelQuantizer from the Quark public API.
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch

from olive.model import HfModelHandler
from olive.passes.pass_config import BasePassConfig

logger = logging.getLogger(__name__)


def run_quark_torch_quantization(
    model: HfModelHandler,
    config: BasePassConfig,
    output_model_path: str,
) -> HfModelHandler:
    """Run Quark 0.11 torch quantization on a HuggingFace model.

    Args:
        model: Olive HfModelHandler pointing to the source model.
        config: Pass configuration with quantization parameters.
        output_model_path: Directory to write quantized model artifacts.

    Returns:
        HfModelHandler pointing to the quantized model output directory.

    """
    # Disable torch dynamo on Windows (Triton not available)
    if sys.platform == "win32":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    from quark.torch import (
        LLMTemplate,
        ModelQuantizer,
        export_gguf,
        export_onnx,
        export_safetensors,
    )
    from quark.torch.utils.llm import (
        get_calib_dataloader,
        get_model,
        get_tokenizer,
        prepare_for_moe_quant,
        revert_model_patching,
    )

    output_dir = Path(output_model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    logger.info("[INFO] Loading model from: %s", model.model_path)
    is_gpu = torch.cuda.is_available()
    torch_model, _ = get_model(
        str(model.model_path),
        config.data_type,
        device,
        multi_gpu=is_gpu,
        multi_device=is_gpu,
        attn_implementation="eager",
        trust_remote_code=config.trust_remote_code,
    )

    prepare_for_moe_quant(torch_model)

    model_type = (
        torch_model.config.model_type
        if hasattr(torch_model.config, "model_type")
        else torch_model.config.architectures[0]
    )

    tokenizer = get_tokenizer(
        str(model.model_path),
        max_seq_len=config.seq_len,
        model_type=model_type,
        trust_remote_code=config.trust_remote_code,
    )

    # 2. Prepare calibration data
    logger.info("[INFO] Loading calibration dataset: %s", config.dataset)
    main_device = torch_model.device
    calib_dataloader = get_calib_dataloader(
        dataset_name=config.dataset,
        processor=None,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_calib_data=config.num_calib_data,
        seqlen=config.seq_len,
        device=main_device,
    )

    # 3. Build quantization config via LLMTemplate
    logger.info("[INFO] Setting up quantization with scheme: %s", config.quant_scheme)

    if model_type not in LLMTemplate.list_available():
        raise ValueError(
            f"Model type '{model_type}' is not supported by LLMTemplate. Available: {LLMTemplate.list_available()}"
        )

    template = LLMTemplate.get(model_type)

    # Handle quant_algo as string or list
    quant_algo_list = _parse_quant_algo(config.quant_algo)

    # Build layer-specific config
    layer_config = dict(config.layer_quant_scheme) if config.layer_quant_scheme else {}

    quant_config = template.get_config(
        scheme=config.quant_scheme,
        algorithm=quant_algo_list,
        kv_cache_scheme=config.kv_cache_dtype,
        min_kv_scale=config.min_kv_scale,
        layer_config=layer_config if layer_config else None,
        attention_scheme=config.attention_dtype,
        exclude_layers=config.exclude_layers,
    )

    # Handle kv_cache_post_rope flag
    if config.kv_cache_post_rope:
        if hasattr(quant_config, "kv_cache_post_rope"):
            quant_config.kv_cache_post_rope = True
        else:
            logger.warning("kv_cache_post_rope not supported by this quant_config")

    # 4. Quantize model
    logger.info("[INFO] Starting model quantization")
    quantizer = ModelQuantizer(quant_config, multi_device=is_gpu)
    torch_model = quantizer.quantize_model(torch_model, calib_dataloader)

    # 5. Freeze model
    logger.info("[INFO] Freezing quantized model")
    torch_model = quantizer.freeze(torch_model)

    # 6. Revert model patching
    logger.info("[INFO] Reverting model patching")
    revert_model_patching(torch_model)

    # 7. Validate export configuration
    if config.custom_mode != "quark" and config.export_weight_format == "fake_quantized":
        raise ValueError("'fake_quantized' export is only supported with custom_mode='quark'")

    # 8. Export model
    logger.info("[INFO] Exporting quantized model to: %s", output_dir)

    export_formats = config.model_export
    if isinstance(export_formats, str):
        export_formats = [export_formats]
    elif export_formats is None:
        export_formats = ["hf_format"]

    for export_format in export_formats:
        if export_format == "hf_format":
            with torch.no_grad():
                export_safetensors(
                    model=torch_model,
                    output_dir=str(output_dir),
                    custom_mode=config.custom_mode,
                    weight_format=config.export_weight_format,
                    pack_method=config.pack_method,
                )
                tokenizer.save_pretrained(str(output_dir))
            logger.info("[INFO] Exported HF format to: %s", output_dir)

            # Workaround for QUARK-476: strip auto_map from exported config.json
            # when trust_remote_code is False, since Quark preserves it from the
            # original model but doesn't copy the referenced .py files.
            if not config.trust_remote_code:
                _strip_auto_map(output_dir)

        elif export_format == "onnx":
            with torch.inference_mode():
                batch_iter = iter(calib_dataloader)
                input_args = next(batch_iter)
                uint4_int4_flag = "uint4" in config.quant_scheme or "int4" in config.quant_scheme

                onnx_output_dir = output_dir / "onnx"
                onnx_output_dir.mkdir(exist_ok=True)
                export_onnx(
                    model=torch_model,
                    output_dir=str(onnx_output_dir),
                    input_args=input_args,
                    uint4_int4_flag=uint4_int4_flag,
                )
            logger.info("[INFO] Exported ONNX format to: %s", onnx_output_dir)

        elif export_format == "gguf":
            with torch.inference_mode():
                export_gguf(
                    torch_model,
                    output_dir=str(output_dir),
                    model_type=model_type,
                    tokenizer_path=str(model.model_path),
                )
            logger.info("[INFO] Exported GGUF format to: %s", output_dir)

    return HfModelHandler(str(output_dir))


def _parse_quant_algo(quant_algo):
    """Normalize quant_algo to a list of strings, or None."""
    if quant_algo is None:
        return None
    if isinstance(quant_algo, list):
        return quant_algo
    if isinstance(quant_algo, str):
        return quant_algo.split(",") if "," in quant_algo else [quant_algo]
    return None


def _strip_auto_map(output_dir: Path):
    """Remove auto_map from exported config.json (QUARK-476 workaround).

    Quark's export_safetensors preserves auto_map from the original model's
    config.json but doesn't copy the referenced auxiliary .py files (e.g.,
    configuration_phi3.py, modeling_phi3.py) to the output directory. This
    causes downstream consumers like onnxruntime_genai builder to fail.

    When trust_remote_code is False, auto_map is unnecessary since
    transformers loads the model natively via model_type.
    """
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config_data = json.load(f)

    if "auto_map" not in config_data:
        return

    auto_map = config_data.pop("auto_map")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info("[INFO] Stripped auto_map from config.json (QUARK-476 workaround): %s", list(auto_map.keys()))
