# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import questionary

from olive.cli.init.helpers import DiffuserVariant, SourceType, _ask, _ask_select

# Diffusers operations
OP_EXPORT = "export"
OP_LORA = "lora"

# Training
TRAIN_STEPS_CUSTOM = "custom"


def run_diffusers_flow(model_config):
    model_path = model_config.get("model_path", "")
    variant = model_config.get("variant", DiffuserVariant.AUTO)

    operation = _ask_select(
        "What do you want to do?",
        choices=[
            questionary.Choice("Export to ONNX (for deployment with ONNX Runtime)", value=OP_EXPORT),
            questionary.Choice("LoRA Training (fine-tune on custom images)", value=OP_LORA),
        ],
    )

    if operation == OP_EXPORT:
        return _export_flow(model_path, variant)
    elif operation == OP_LORA:
        return _lora_flow(model_path, variant)
    return {}


def _export_flow(model_path, variant):
    torch_dtype = _ask(
        questionary.select(
            "Torch dtype:",
            choices=[
                questionary.Choice("float16", value="float16"),
                questionary.Choice("float32", value="float32"),
            ],
        )
    )

    cmd = f"olive capture-onnx-graph -m {model_path} --torch_dtype {torch_dtype}"
    if variant != DiffuserVariant.AUTO:
        cmd += f" --model_variant {variant}"

    return {"command": cmd}


def _lora_flow(model_path, variant):
    # LoRA parameters
    lora_r = _ask(
        questionary.select(
            "LoRA rank (r):",
            choices=[
                questionary.Choice("16 (recommended)", value="16"),
                questionary.Choice("4", value="4"),
                questionary.Choice("8", value="8"),
                questionary.Choice("32", value="32"),
                questionary.Choice("64", value="64"),
            ],
        )
    )

    lora_alpha = _ask(
        questionary.text(
            "LoRA alpha (default = same as rank):",
            default=lora_r,
        )
    )

    lora_dropout = _ask(questionary.text("LoRA dropout:", default="0.0"))

    # Data source
    data_source = _ask(
        questionary.select(
            "Training data source:",
            choices=[
                questionary.Choice("Local image folder", value=SourceType.LOCAL),
                questionary.Choice("HuggingFace dataset", value=SourceType.HF),
            ],
        )
    )

    data_args = []
    if data_source == SourceType.LOCAL:
        data_dir = _ask(
            questionary.text(
                "Path to image folder:",
                validate=lambda x: True if x.strip() else "Please enter a path",
            )
        )
        data_args.append(f"-d {data_dir}")
    else:
        data_name = _ask(
            questionary.text(
                "Dataset name:",
                instruction="e.g., linoyts/Tuxemon",
            )
        )
        data_split = _ask(questionary.text("Split:", default="train"))
        image_column = _ask(questionary.text("Image column name:", default="image"))
        caption_column = _ask(questionary.text("Caption column name (optional):", default=""))

        data_args.extend(
            [
                f"--data_name {data_name}",
                f"--data_split {data_split}",
                f"--image_column {image_column}",
            ]
        )
        if caption_column:
            data_args.append(f"--caption_column {caption_column}")

    # DreamBooth
    dreambooth_args = []
    enable_dreambooth = _ask(questionary.confirm("Enable DreamBooth training?", default=False))
    if enable_dreambooth:
        dreambooth_args.append("--dreambooth")

        instance_prompt = _ask(
            questionary.text(
                "Instance prompt (e.g., 'a photo of sks dog'):",
                validate=lambda x: True if x.strip() else "Instance prompt is required for DreamBooth",
            )
        )
        dreambooth_args.append(f'--instance_prompt "{instance_prompt}"')

        with_prior = _ask(questionary.confirm("Enable prior preservation?", default=True))
        if with_prior:
            class_prompt = _ask(
                questionary.text(
                    "Class prompt (e.g., 'a photo of a dog'):",
                    validate=lambda x: True if x.strip() else "Class prompt is required for prior preservation",
                )
            )
            dreambooth_args.extend(
                [
                    "--with_prior_preservation",
                    f'--class_prompt "{class_prompt}"',
                ]
            )

            class_data_dir = _ask(questionary.text("Class data directory (optional):", default=""))
            if class_data_dir:
                dreambooth_args.append(f"--class_data_dir {class_data_dir}")

            num_class_images = _ask(questionary.text("Number of class images:", default="200"))
            dreambooth_args.append(f"--num_class_images {num_class_images}")

    # Training parameters
    max_train_steps = _ask(
        questionary.select(
            "Max training steps:",
            choices=[
                questionary.Choice("1000 (recommended)", value="1000"),
                questionary.Choice("500 (quick)", value="500"),
                questionary.Choice("2000 (thorough)", value="2000"),
                questionary.Choice("Custom", value=TRAIN_STEPS_CUSTOM),
            ],
        )
    )
    if max_train_steps == TRAIN_STEPS_CUSTOM:
        max_train_steps = _ask(questionary.text("Enter max training steps:"))

    learning_rate = _ask(questionary.text("Learning rate:", default="1e-4"))
    train_batch_size = _ask(questionary.text("Train batch size:", default="1"))
    gradient_accumulation = _ask(questionary.text("Gradient accumulation steps:", default="4"))

    mixed_precision = _ask(
        questionary.select(
            "Mixed precision:",
            choices=[
                questionary.Choice("bf16 (recommended)", value="bf16"),
                questionary.Choice("fp16", value="fp16"),
                questionary.Choice("no", value="no"),
            ],
        )
    )

    lr_scheduler = _ask(
        questionary.select(
            "Learning rate scheduler:",
            choices=[
                questionary.Choice("constant", value="constant"),
                questionary.Choice("linear", value="linear"),
                questionary.Choice("cosine", value="cosine"),
                questionary.Choice("cosine_with_restarts", value="cosine_with_restarts"),
                questionary.Choice("polynomial", value="polynomial"),
                questionary.Choice("constant_with_warmup", value="constant_with_warmup"),
            ],
        )
    )

    warmup_steps = _ask(questionary.text("Warmup steps:", default="0"))
    seed = _ask(questionary.text("Random seed (optional, press Enter to skip):", default=""))

    # Flux-specific
    flux_args = []
    if variant == DiffuserVariant.FLUX:
        guidance_scale = _ask(questionary.text("Guidance scale (Flux-specific):", default="3.5"))
        flux_args.append(f"--guidance_scale {guidance_scale}")

    # Merge LoRA
    merge_lora = _ask(questionary.confirm("Merge LoRA into base model?", default=False))

    # Build command
    cmd_parts = ["olive diffusion-lora", f"-m {model_path}"]
    if variant != DiffuserVariant.AUTO:
        cmd_parts.append(f"--model_variant {variant}")
    cmd_parts.extend(
        [
            f"-r {lora_r}",
            f"--alpha {lora_alpha}",
            f"--lora_dropout {lora_dropout}",
            *data_args,
            *dreambooth_args,
            f"--max_train_steps {max_train_steps}",
            f"--learning_rate {learning_rate}",
            f"--train_batch_size {train_batch_size}",
            f"--gradient_accumulation_steps {gradient_accumulation}",
            f"--mixed_precision {mixed_precision}",
            f"--lr_scheduler {lr_scheduler}",
            f"--lr_warmup_steps {warmup_steps}",
        ]
    )
    if seed:
        cmd_parts.append(f"--seed {seed}")
    cmd_parts.extend(flux_args)
    if merge_lora:
        cmd_parts.append("--merge_lora")

    cmd = " ".join(cmd_parts)

    return {"command": cmd}
