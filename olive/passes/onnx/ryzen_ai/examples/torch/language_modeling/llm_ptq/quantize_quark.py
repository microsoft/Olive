#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
import os
from pathlib import Path
import torch
import warnings
import argparse

from quark.torch import ModelQuantizer, ModelImporter, ModelExporter, load_params, save_params

from olive.passes.onnx.ryzen_ai.examples.torch.language_modeling.llm_ptq.customized_configuration import SUPPORTED_QUANT_SCHEME
from olive.passes.onnx.ryzen_ai.examples.torch.language_modeling.llm_ptq.configuration_preparation import get_config, get_export_config
from quark.torch.export.api import _move_quantizer_to_dict
from quark.torch.utils.device import TPDeviceManager
from transformers import AutoProcessor

# TODO: Using sys.path.append is bad practice.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from olive.passes.onnx.ryzen_ai.examples.torch.language_modeling.llm_utils.data_preparation import get_calib_dataloader
from olive.passes.onnx.ryzen_ai.examples.torch.language_modeling.llm_eval.evaluation import eval_model
from olive.passes.onnx.ryzen_ai.examples.torch.language_modeling.llm_utils.model_preparation import get_model, get_model_type, get_tokenizer, prepare_for_moe_quant


def run_quark_quantization(args: argparse.Namespace) -> None:
    # 1. Define original model
    print("\n[INFO]: Loading model ...")

    # We currently use CPU memory to load large models because GPU memory is typically smaller.
    # The model will be dispatched to different GPUs based on the total number of GPUs specified by torchrun --nproc-per-node.
    # TODO:
    # The current method results in high CPU memory consumption due to multiple copies of the same model.
    # We plan to address this in the future by implementing a more efficient way to dispatch the model to devices.
    if args.use_tp:
        device = "cpu"
    else:
        device = args.device

    model, model_dtype = get_model(args.model_dir, args.data_type, device, args.multi_gpu, args.multi_device, args.model_attn_implementation)
    prepare_for_moe_quant(model)

    model_type = get_model_type(model)
    tokenizer = get_tokenizer(args.model_dir, max_seq_len=args.seq_len, model_type=model_type)
    multimodal = True if model_type in ["mllama"] else False
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
        print("\nRestore quantized model from json and safetensors file ...")
        model = load_params(model, json_path=args.json_path, safetensors_path=args.safetensors_path)
        args.skip_quantization = True
    elif args.model_reload:
        print(f"\nRestore quantized model from {args.import_file_format} file ...")

        importer = ModelImporter(model_info_dir=args.import_model_dir, saved_format=args.import_file_format, multi_device=args.multi_device)
        model = importer.import_model_info(model)

        args.skip_quantization = True

    if args.use_tp:
        if TPDeviceManager._tp_mesh is not None:
            _move_quantizer_to_dict(model.model)

            device = TPDeviceManager._device
            tp_mesh = TPDeviceManager._tp_mesh

            model.tensor_parallel(tp_mesh)
            model.to(device)
        else:
            warnings.warn(
                "Quark tensor parallelism is not initialized properly. Please check the torchrun settings.",
                UserWarning)
            return

    # 3. Define calibration dataloader(still need this step for weight only and dynamic quantization in Quark for current version.)
    print("\n[INFO]: Loading dataset ...")
    # When the model is small, accelerate will place it on the last device
    main_device = model.device if args.multi_gpu or args.multi_device else args.device
    calib_dataloader = get_calib_dataloader(dataset_name=args.dataset,
                                            processor=processor if multimodal else None,
                                            tokenizer=tokenizer,
                                            batch_size=args.batch_size,
                                            num_calib_data=args.num_calib_data,
                                            seqlen=args.seq_len,
                                            device=main_device)

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
            print("\n[INFO]: Exporting quark native json and pth...")
            with torch.no_grad():
                quant_config = get_config(args, model_type)
                exporter.export_quark_model(model, quant_config=quant_config, custom_mode=args.custom_mode)

        # Export option 2: hugging-face safetensors format
        if "hf_format" in args.model_export:
            print("\n[INFO]: Exporting hugging face format safetensors...")
            with torch.no_grad():
                quant_config = get_config(args, model_type)
                exporter.export_safetensors_model(
                    model,
                    quant_config=quant_config,
                    custom_mode=args.custom_mode,
                    tokenizer=tokenizer
                )

        # Export option 3: onnx
        if "onnx" in args.model_export:
            print("\n[INFO]: Exporting onnx graph...")
            with torch.inference_mode():
                batch_iter = iter(calib_dataloader)
                input_args = next(batch_iter)
                if args.quant_scheme in ["w_int4_per_channel_sym", "w_uint4_per_group_asym", "w_int4_per_group_sym", "w_uint4_a_bfloat16_per_group_asym"]:
                    uint4_int4_flag = True
                else:
                    uint4_int4_flag = False

                exporter.export_onnx_model(model, input_args, uint4_int4_flag=uint4_int4_flag)
        # Export option 3: gguf
        if "gguf" in args.model_export:
            print("\n[INFO]: Exporting gguf model...")
            with torch.inference_mode():
                exporter.export_gguf_model(model, args.model_dir, model_type)

    # 7. (Optional) Torch compile
    if args.torch_compile:
        print("\n[INFO]: Calling PyTorch 2 torch.compile...")
        # Note: The model after torch.compile may not be able to export to other format
        model = torch.compile(model)

    # 8. (Optional) Model Parameters Save
    if args.params_save:
        save_params(model, model_type=model_type, export_dir=args.save_dir)

    # 9. (Optional) Model Evaluation
    if not args.skip_evaluation:
        print("\n[INFO]: Evaluating ...")
        eval_model(args, model, main_device, save_metrics_to_csv=args.save_metrics_to_csv, output_dir=args.metrics_output_dir, multimodal=multimodal)

    if args.use_tp:
        TPDeviceManager.tp_cleanup()
        
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Argument for model
    parser.add_argument("--model_dir", help="Specify where the HuggingFace model is. This example support Llama, OPT models", required=True)
    parser.add_argument("--device", help="Device for running the quantizer", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument("--model_attn_implementation", help="The attention implementation to use in the model", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--multi_device",
                        action="store_true",
                        help="we allow you to use this mode to run a model quantization that exceeds the size of your gpu memory if you use args.multi_gpu and still run into OOM "
                             "now it only supports thr common quantization without algorithms, please note that this can lead to very slow quantization.")

    # Argument for calibration dataset
    parser.add_argument("--dataset", help="Dataset for calibration", default="pileval", choices=["pileval", "wikitext", "pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark", "HuggingFaceH4/ultrachat_200k", "ScienceQA"])
    parser.add_argument("--data_type", help="Datatype of the model", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--seq_len", type=int, help="Sequence length of data", default=512)
    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument("--num_calib_data", help="Number of samples for calibration.", type=int, default=512)

    # Argument for quantization
    parser.add_argument("--skip_quantization", action='store_true')
    parser.add_argument("--group_size", help="Group size for per_group quantization.", type=int, default=128)
    parser.add_argument(
        "--group_size_per_layer",
        action="append",
        nargs=2,
        metavar=("PATTERN", "GROUP_SIZE"),
        help="Set a specific group size for layers matching the given pattern. This argument can be repeated for multiple patterns. "
        "Usage: `--group_size_per_layer lm_head 32`.",
    )
    parser.add_argument("--quant_scheme",
                        help="Supported quant_scheme in the script. If there is no suitable quantization strategy among the options, users can customize the quantization configuration according to their own needs.", default=None, choices=SUPPORTED_QUANT_SCHEME
                        )
    parser.add_argument("--kv_cache_dtype", "--kv_cache_quant_scheme", help="KV Cache dtype.", default=None, choices=["fp8", "fp8_dynamic", "int8_per_tensor_static", "int8_per_tensor_dynamic", "int8_per_token", "mxfp8", "fp6e2m3_per_group", "mxfp6_e2m3", "fp6e3m2_per_group", "mxfp6_e3m2", "fp4_per_group", "mxfp4", None])
    parser.add_argument("--min_kv_scale", help="Minimum value of KV Cache scale.", type=float, default=0.0)
    parser.add_argument("--pre_quantization_optimization", help="Pre Quantization Optimization.", choices=["rotation", "smoothquant", "quarot"], action='append', default=[])
    parser.add_argument("--pre_optimization_config_file_path", help="The JSON file path of pre-optimization config.", type=str, default=None)
    parser.add_argument("--quant_algo", help="Quantization Algorithms.", default=None, choices=["awq", "gptq", "autosmoothquant", "quarot", None])
    parser.add_argument("--quant_algo_config_file_path", help="The JSON file path of quantization algorithm config.", type=str, default=None)
    parser.add_argument('--exclude_layers', type=str,
                        nargs='*',  # Allows to pass a list of strings
                        default=None,  # Default is None to allow model-specific layer exclusion
                        help='List of layers to exclude from quantization. Default depends on model type. Usage: `--exclude_layers "*down_proj*" "*31.fc*" "*k_proj"`. To avoid excluding layers at all, simply use `--exclude_layers` without any argument.')
    parser.add_argument("--scale_format", help="Scale format", default="e4m3", choices=["e4m3", "float32"])
    parser.add_argument("--scale_calculation_mode", help="Scale calculation mode", default="even", choices=["even", "floor", "ceil"])

    # Argument for custom quantization
    parser.add_argument("--fp8_attention_quant", action="store_true", help="Enable fp8 attention quantization")
    parser.add_argument("--moe_experts_second_step_config", help="The second step quantization config for MoE experts weights.", type=str, default=None, choices=["w_int4_per_channel_sym"])

    # Argument for reloading
    parser.add_argument("--model_reload", help="safetensors or pth model reload", action="store_true")
    parser.add_argument("--import_model_dir", help="directory of hf or quark model")
    parser.add_argument("--params_load", help="Model parameters load", action="store_true")
    parser.add_argument("--json_path", help="Specify the path of saved json file")
    parser.add_argument("--safetensors_path", help="Specify the path of saved safetensors file")
    parser.add_argument("--import_file_format", type=str, help="file_format for importing. If you export hf_format, you should use 'hf_format' for reloading.", default="quark_format", choices=["quark_format", "hf_format"])

    # Argument for export
    parser.add_argument("--model_export", help="Model export format", default=None, action="append", choices=[None, "onnx", "quark_format", "hf_format", "gguf"])
    parser.add_argument("--custom_mode", help="When selecting `--custom_mode awq` or `--custom_mode fp8`, this legacy argument allows to export FP8 and AWQ models in the custom format they were exported with with quark<1.0, with custom config saved in the config.json, and config checkpoint format (AWQ uses `qzeros`, `qweight`, transposed `scales`).", default="quark", type=str, choices=["quark", "awq", "fp8"])
    parser.add_argument("--torch_compile", help="Model torch compile", action="store_true")
    parser.add_argument("--pack_method", type=str, help="Pack method for awq_export", default="reorder", choices=["order", "reorder"])
    parser.add_argument("--output_dir", default="exported_model")

    parser.add_argument("--weight_matrix_merge", help="Whether to merge weight matrix when dump llm-specific quantized model", action='store_true')
    parser.add_argument("--export_weight_format", type=str, help="Whether to export weights compressed or uncompressed", default="real_quantized", choices=["fake_quantized", "real_quantized"])

    # Argument for saving
    parser.add_argument("--params_save", help="Model parameters save", action='store_true')
    parser.add_argument("--save_dir", help="Directory to save model parameters as safetensors or pth, in the case when --params_save is used.", default="model_params")

    # Argument for evaluation
    parser.add_argument("--skip_evaluation", action='store_true')
    parser.add_argument("--save_metrics_to_csv", action="store_true")
    parser.add_argument("--metrics_output_dir", default="metrics_output_dir", help="Output path of csv with metrics.")
    parser.add_argument("--tasks", default=None, type=str, metavar="task1,task2", help="Comma-separated list of task names or task groupings to evaluate on.")
    parser.add_argument("--use_ppl_eval_for_kv_cache", action="store_true")
    parser.add_argument("--ppl_eval_for_kv_cache_context_size", type=int, help="Context size used in PPL evaluation for KV cache.", default=1024)
    parser.add_argument("--ppl_eval_for_kv_cache_sample_size", type=int, help="Sample size used in PPL evaluation for KV cache.", default=512)
    parser.add_argument("--ppl_eval_for_kv_cache_patch_size", type=int, help="Patch size used in PPL evaluation for KV cache.", default=None)
    parser.add_argument("--eval_batch_size", type=str, default=8, metavar="auto|auto:N|N", help="Batch size for evaluation. Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.")
    parser.add_argument("--max_eval_batch_size", type=int, default=None, metavar="N", help="Maximal batch size to try with --batch_size auto.")
    parser.add_argument("--num_eval_data", help="Number of samples for evaluation. The default value is -1, which means the entire dataset is used for evaluation.", type=int, default=-1)
    parser.add_argument("--num_fewshot", type=int, default=None, metavar="N", help="Number of examples in few-shot context")
    parser.add_argument("--apply_chat_template", action="store_true", help="Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt.")
    parser.add_argument("--use_mlperf_rouge", action="store_true")
    parser.add_argument("--eval_data_dir", help="Dataset for evaluation", type=str, default=None)
    parser.add_argument("--use_tp", action="store_true", help="Enable tensor parallelism exclusively for model evaluation.")

    args = parser.parse_args()

    run_quark_quantization(args)

    

if __name__ == "__main__":
    main()
    