# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# mapping from task to peft task type
# refer to peft.utils.peft_types.TaskType for all possible values
TASK_TO_PEFT_TASK_TYPE = {
    "text-classification": "SEQ_CLS",
    "text-generation": "CAUSAL_LM",
    # TODO(jambayk): see if we need more task types
}

# model_type -> max length of the model, extracted from the config
# will be int if not present in the config
MODELS_TO_MAX_LENGTH_MAPPING = {
    "__default__": "max_position_embeddings",
    "bloom": 2048,
    "gpt2": "n_positions",
    "gpt_neox": "max_position_embeddings",
    "gptj": "n_postions",
    "llama": "max_position_embeddings",
    "mistral": "max_position_embeddings",
    "opt": "max_position_embeddings",
    "phi": "max_position_embeddings",
    "phi3": "max_position_embeddings",
    "qwen": "seq_length",
    "qwen2": "max_position_embeddings",
}

MODEL_TYPE_MAPPING = {
    "whisper": "bart",
    "camembert": "bert",
    "deberta": "bert",
    "deberta-v2": "bert",
    "distilbert": "bert",
    "gpt_neox": "gpt2",
    "gpt-j": "gpt2",
    "llama": "gpt2",
    "roberta": "bert",
    "phi3": "phi",
}

MODELS_TO_LORA_TARGET_MODULES_MAPPING = {"phi3": ["o_proj", "qkv_proj"]}
