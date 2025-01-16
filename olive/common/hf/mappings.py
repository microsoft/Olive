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
