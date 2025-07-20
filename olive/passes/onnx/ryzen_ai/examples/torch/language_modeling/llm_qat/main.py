#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, HfArgumentParser
from accelerate import Accelerator

# TODO: using sys.path.append is bad practice.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm_eval.evaluation import task_eval
from llm_utils.data_preparation import get_loader, get_trainer_dataset

from util.api import weight_only_quantize

from quark.torch.export import ExporterConfig, JsonExporterConfig
from quark.torch.export.api import ModelExporter, ModelImporter
from quark.torch.export.safetensors import _load_weights_from_safetensors


from dataclasses import dataclass, field
from pprint import pformat
import evaluate
from typing import Optional


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


@dataclass
class DataArguments:
    finetune_dataset: str = field(default="wikitext", metadata={"choices": ["wikitext", "shibing624/AdvertiseGen"]})
    max_train_samples: Optional[int] = field(default=None, metadata={'help': "It will sample max_train_samples from train dataset"})
    max_eval_samples: Optional[int] = field(default=None, metadata={'help': "It will sample max_eval_samples from eval dataset"})


@dataclass
class TrainingArguments(TrainingArguments):
    model: Optional[str] = field(default="THUDM/chatglm3-6b", metadata={"help": "Specify where the HuggingFace model is."})
    model_trust_remote_code: bool = field(default=False)
    skip_quantization: bool = field(default=False)
    quant_resume: bool = field(default=False)
    quant_scheme: str = field(
                default="w_uint4_asym",
                metadata={
                    "help": ("Supported quant_scheme in the script."
                             "If there is no suitable quantization strategy among the options,"
                             "users can customize the quantization configuration according to their own needs."
                             "If None, the model will be quantized by float16"),
                    "choices": ["w_uint4_asym", "w_int4_sym"]
                }
    )
    finetune_seqlen: int = field(default=512)
    group_size: int = field(
        default=128, metadata={'help': "Group size for per_group quantization."}
    )
    kv_cache_dtype: Optional[str] = field(
        default=None, metadata={"help": "KV Cache dtype.", "choices": ["fp8", None]}
    )
    skip_finetune: bool = field(default=False)
    custom_mode: str = field(
        default="quark", metadata={"help": "When selecting `--custom_mode awq` or `--custom_mode fp8`, this legacy argument allows to export FP8 and AWQ models in the custom format they were exported with with quark<1.0, with custom config saved in the config.json, and config checkpoint format (AWQ uses `qzeros`, `qweight`, transposed `scales`).", "choices": ["quark", "awq", "fp8"]}
    )
    pack_method: str = field(
        default="reorder", metadata={"help": "Pack method for awq_export", "choices": ["order", "reorder"]}
    )
    weight_matrix_merge: bool = field(
        default=False, metadata={'help': "Whether to merge weight matrix when dump llm-specific quantized model"}
    )
    model_reload: bool = field(
        default=False, metadata={"help": "Safetensors or pth model reload"}
    )
    import_file_format: str = field(
        default="hf_format", metadata={"help": "file_format for importing. If you export hf_format, you should use 'hf_format' for reloading.", "choices": ["quark_format", "hf_format"]}
    )
    import_model_dir: Optional[str] = field(
        metadata={"help": "directory of hf or quark model"}, default=None
    )
    skip_evaluation: bool = field(default=False)
    eval_task: str = field(
        metadata={"help": "Comma-separated list of task names or task groupings to evaluate on."}, default='wikitext'
    )
    max_eval_batch_size: Optional[int] = field(
        default=None, metadata={'help': "Maximal batch size to try with --batch_size auto."}
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)
    eval_result_output_path: Optional[str] = field(default=None, metadata={"help": "lm eval output result path"})
    eval_strategy: str = field(default='epoch')
    save_strategy: str = field(default='epoch')
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    save_only_model: bool = field(default=False)
    metric_for_best_model: str = field(default='eval_loss')
    gradient_checkpointing: bool = field(default=False)
    logging_strategy: str = field(default='epoch')
    attn_implementation: str = field(default='eager')
    report_to: Optional[str] = field(default="none")


@dataclass
class ExportArguments:
    model_export: Optional[str] = field(
        default=None, metadata={"help": "Model export format.", "choices": ["onnx", "quark_format", "hf_format", "gguf", None]}
    )
    model_export_dir: str = field(default="exported_model")
    export_weight_format: str = field(
        default="real_quantized", metadata={"help": "Whether to export weights compressed or uncompressed", "choices": ["fake_quantized", "real_quantized"]}
    )


def print_training_args(args: TrainingArguments, exclude_defaults=True):
    args_dict = vars(args).copy()
    if exclude_defaults:
        default_args = TrainingArguments("tmp_output_dir")
        default_dict = vars(default_args)

        # Filtering unmodifed training arguments
        filtered_dict = {
            k: v for k, v in args_dict.items()
            if str(v) != str(default_dict.get(k, None))
        }
        args_dict = filtered_dict

    banner = "=" * 40 + " Arguments " + "=" * 40
    formatted = pformat(args_dict, indent=2, width=120, sort_dicts=False)

    try:
        from termcolor import colored
        accelerator.print(colored(banner, 'cyan'))
        accelerator.print(colored(formatted, 'yellow'))
    except ImportError:
        accelerator.print(banner)
        accelerator.print(formatted)


def load_fsdp_full_state_model(training_args, export_args):
    model, tokenizer = load_original_model(training_args)
    calib_loader = None
    model, quant_config = weight_only_quantize(model, calib_loader, training_args.quant_scheme, training_args.group_size)
    model_state_dict = _load_weights_from_safetensors(export_args.model_export_dir)
    model.load_state_dict(model_state_dict)
    return model


def load_original_model(training_args):
    accelerator.print("\n[QUARK-INFO]: Loading Model and Tokenizer... ")
    torch_dtype = "auto" if training_args.skip_finetune else torch.bfloat16

    model_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": training_args.model_trust_remote_code, "attn_implementation": training_args.attn_implementation}
    model = AutoModelForCausalLM.from_pretrained(training_args.model, **model_kwargs)

    tokenizer_kwargs = {"trust_remote_code": training_args.model_trust_remote_code}
    tokenizer = AutoTokenizer.from_pretrained(training_args.model, **tokenizer_kwargs)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # ChatGLM model do not have bos_token
    if 'sop' in tokenizer.get_added_vocab() and not tokenizer.bos_token:
        tokenizer.add_special_tokens({
            "bos_token": "sop",
        })
        embeddings = model.get_input_embeddings()
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def run(training_args, data_args, export_args):
    # 1. Define original model
    model, tokenizer = load_original_model(training_args)

    # 2. (Optional) Reload quantized model that is exported by Quark
    if training_args.model_reload:
        accelerator.print("\nRestore quantized model from hf_format file ...")
        importer = ModelImporter(model_info_dir=training_args.import_model_dir, saved_format="safetensors")
        model = importer.import_model_info(model)
        training_args.skip_quantization = True

    # 3. PTQ and load checkpoint (optional)
    if not training_args.skip_quantization:
        accelerator.print("\n[QUARK-INFO]: Quantizing... ")
        calib_loader = get_loader('wikitext', 'test', tokenizer, seqlen=2048, num_batch=1)
        model, quant_config = weight_only_quantize(model, calib_loader, training_args.quant_scheme, training_args.group_size)

        if training_args.quant_resume:
            model_state_file = os.path.join(training_args.finetune_checkpoint, 'best.pth')
            state = torch.load(model_state_file, weights_only=True, map_location="cuda")
            model.load_state_dict(state['state_dict'])
            accelerator.print(f"\n[QUARK-INFO]: ReLoaded checkpoint from {model_state_file}")

    # 4. Finetuning
    if not training_args.skip_finetune:
        accelerator.print("\n Trainer Fine-Tuning... ")
        data_module = get_trainer_dataset(data_args.finetune_dataset, 'train', tokenizer, data_args.max_train_samples, data_args.max_eval_samples, seqlen=training_args.finetune_seqlen)

        # Training
        # Torch >= 2.4 throws an error if `use_reentrant` is not set explicitly
        if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

        trainer = Trainer(model=model,
                          tokenizer=tokenizer,
                          args=training_args,
                          compute_metrics=compute_metrics,
                          preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                          **data_module)
        trainer._move_model_to_device(model, trainer.args.device)
        trainer.train()

        # Obtain FSDP Model for further modification
        if hasattr(trainer.model, "module"):
            # Prepare for obtaining full state dict under FSDP training setting
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
            trainer.save_model(export_args.model_export_dir)
            accelerator.wait_for_everyone()

            accelerator.print('Loading FSDP Full state model ...')
            model = load_fsdp_full_state_model(training_args, export_args)
            torch.cuda.empty_cache()
            accelerator.print('Loading FSDP Full state done!')

        else:
            model = trainer.model

        accelerator.wait_for_everyone()


    # 5. Export safetensors
    # If load_best_model_at_end, it will export the best model
    if export_args.model_export is not None:
        if trainer.is_world_process_zero():
            os.makedirs(export_args.model_export_dir, exist_ok=True)
            json_export_config = JsonExporterConfig(weight_format=export_args.export_weight_format, pack_method="reorder")
            export_config = ExporterConfig(json_export_config=json_export_config)
            exporter = ModelExporter(config=export_config, export_dir=export_args.model_export_dir)

            with torch.no_grad():
                exporter.export_safetensors_model(model, quant_config=quant_config, custom_mode="quark", tokenizer=tokenizer)

    accelerator.wait_for_everyone()

    # 6. Evaluation
    if not training_args.skip_evaluation:
        model.eval()
        model.to('cuda')
        dtype = training_args.quant_scheme if not training_args.skip_quantization else str(next(model.parameters()).dtype)
        accelerator.print(f"\n[QUARK-INFO]: Evaluating ({dtype})... ")
        with torch.no_grad():
            num_fewshot, apply_chat_template = None, False
            task_eval(model, tokenizer, training_args.per_device_eval_batch_size, training_args.max_eval_batch_size, training_args.eval_task, num_fewshot, apply_chat_template, output_path=training_args.eval_result_output_path)


if __name__ == '__main__':
    trainer_parser = HfArgumentParser(
        (TrainingArguments, DataArguments, ExportArguments)
    )
    accelerator = Accelerator()
    training_args, data_args, export_args = trainer_parser.parse_args_into_dataclasses()

    msg = '\n'.join([f'{k:<26}: {v}' for k, v in vars(data_args).items()])
    accelerator.print(f"\n{msg}")

    print_training_args(training_args)
    metric = evaluate.load("accuracy")
    run(training_args, data_args, export_args)
