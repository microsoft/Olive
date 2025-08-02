#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import argparse
import logging
import platform
import warnings
from pathlib import Path

import torch
from quark.torch import ModelExporter, ModelImporter, ModelQuantizer, load_params, save_params
from quark.torch.export.api import _move_quantizer_to_dict
from quark.torch.utils.device import TPDeviceManager
from transformers import AutoProcessor

from olive.passes.quark_quantizer.torch.language_modeling.llm_ptq.configuration_preparation import (
    get_config,
    get_export_config,
)
from olive.passes.quark_quantizer.torch.language_modeling.llm_utils.data_preparation import get_calib_dataloader
from olive.passes.quark_quantizer.torch.language_modeling.llm_utils.model_preparation import (
    get_model,
    get_model_type,
    get_tokenizer,
    prepare_for_moe_quant,
)

logger = logging.getLogger(__name__)


def run_quark_quantization(args: argparse.Namespace) -> None:
    # 1. Define original model
    logger.info("\n[INFO]: Loading model ...")

    # We currently use CPU memory to load large models because GPU memory is typically smaller.
    # The model will be dispatched to different GPUs based on the total number of GPUs specified by torchrun --nproc-per-node.
    # The current method results in high CPU memory consumption due to multiple copies of the same model.
    # We plan to address this in the future by implementing a more efficient way to dispatch the model to devices.
    if args.use_tp or platform.system().lower() == "windows" or not torch.cuda.is_available():
        device = "cpu"
    else:
        device = args.device

    model, _ = get_model(
        args.model_dir, args.data_type, device, args.multi_gpu, args.multi_device, args.model_attn_implementation
    )
    prepare_for_moe_quant(model)

    model_type = get_model_type(model)
    tokenizer = get_tokenizer(args.model_dir, max_seq_len=args.seq_len, model_type=model_type)
    multimodal = model_type in ["mllama"]
    if multimodal:
        processor = AutoProcessor.from_pretrained(args.model_dir)
        if args.model_export is not None:
            export_dir = Path(args.output_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(args.output_dir)

    if args.use_tp:
        TPDeviceManager.tp_mesh_init()

    # 2. (Optional) Reload quantized model
    if args.params_load:
        logger.info("\nRestore quantized model from json and safetensors file ...")
        model = load_params(model, json_path=args.json_path, safetensors_path=args.safetensors_path)
        args.skip_quantization = True
    elif args.model_reload:
        logger.info("\nRestore quantized model from %s file ...", args.import_file_format)

        importer = ModelImporter(
            model_info_dir=args.import_model_dir, saved_format=args.import_file_format, multi_device=args.multi_device
        )
        model = importer.import_model_info(model)

        args.skip_quantization = True

    if args.use_tp:
        if TPDeviceManager._tp_mesh is not None:  # pylint: disable=protected-access
            _move_quantizer_to_dict(model.model)

            device = TPDeviceManager._device  # pylint: disable=protected-access
            tp_mesh = TPDeviceManager._tp_mesh  # pylint: disable=protected-access

            model.tensor_parallel(tp_mesh)
            model.to(device)
        else:
            warnings.warn(
                "Quark tensor parallelism is not initialized properly. Please check the torchrun settings.", UserWarning
            )
            return

    # 3. Define calibration dataloader(still need this step for weight only and dynamic quantization in Quark for current version.)
    logger.info("\n[INFO]: Loading dataset ...")
    # When the model is small, accelerate will place it on the last device
    main_device = model.device if args.multi_gpu or args.multi_device else args.device
    calib_dataloader = get_calib_dataloader(
        dataset_name=args.dataset,
        processor=processor if multimodal else None,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_calib_data=args.num_calib_data,
        seqlen=args.seq_len,
        device=main_device,
    )

    # 4. Quantization
    if not args.skip_quantization:
        # 4-1. Set quantization configuration
        quant_config = get_config(args, model_type)
        # 4-2. In-place replacement of model modules with quantized versions.
        quantizer = ModelQuantizer(quant_config, args.multi_device)
        model = quantizer.quantize_model(model, calib_dataloader)
        args.exclude_layers = quantizer.config.exclude

    # 5. (Optional) Model freeze
    if not args.skip_quantization and (args.model_export is not None or args.params_save or args.torch_compile):
        # If user want to export the quantized model, please freeze the quantized model first
        model = quantizer.freeze(model)

    # 6. (Optional) Model exporting
    if args.model_export is not None:
        export_config = get_export_config(args, model_type)
        if args.custom_mode != "quark" and args.export_weight_format == "fake_quantized":
            raise ValueError("Exporting with 'fake_quantized' only supports custom_mode=quark")
        export_config.json_export_config.weight_format = args.export_weight_format
        exporter = ModelExporter(config=export_config, export_dir=args.output_dir)

        # Export option 1: quark format: native json-pth format
        if "quark_format" in args.model_export:
            if args.custom_mode != "quark":
                raise ValueError("To export the quark_format format, you must use 'args.custom_mode=quark'")
            logger.info("\n[INFO]: Exporting quark native json and pth...")
            with torch.no_grad():
                quant_config = get_config(args, model_type)
                exporter.export_quark_model(model, quant_config=quant_config, custom_mode=args.custom_mode)

        # Export option 2: hugging-face safetensors format
        if "hf_format" in args.model_export:
            logger.info("\n[INFO]: Exporting hugging face format safetensors...")
            with torch.no_grad():
                quant_config = get_config(args, model_type)
                exporter.export_safetensors_model(
                    model, quant_config=quant_config, custom_mode=args.custom_mode, tokenizer=tokenizer
                )

        # Export option 3: onnx
        if "onnx" in args.model_export:
            logger.info("\n[INFO]: Exporting onnx graph...")
            with torch.inference_mode():
                batch_iter = iter(calib_dataloader)
                input_args = next(batch_iter)
                uint4_int4_flag = args.quant_scheme in [
                    "w_int4_per_channel_sym",
                    "w_uint4_per_group_asym",
                    "w_int4_per_group_sym",
                    "w_uint4_a_bfloat16_per_group_asym",
                ]
                exporter.export_onnx_model(model, input_args, uint4_int4_flag=uint4_int4_flag)
        # Export option 3: gguf
        if "gguf" in args.model_export:
            logger.info("\n[INFO]: Exporting gguf model...")
            with torch.inference_mode():
                exporter.export_gguf_model(model, args.model_dir, model_type)

    # 7. (Optional) Torch compile
    if args.torch_compile:
        logger.info("\n[INFO]: Calling PyTorch 2 torch.compile...")
        # Note: The model after torch.compile may not be able to export to other format
        model = torch.compile(model)

    # 8. (Optional) Model Parameters Save
    if args.params_save:
        save_params(model, model_type=model_type, export_dir=args.save_dir)

    if args.use_tp:
        TPDeviceManager.tp_cleanup()
