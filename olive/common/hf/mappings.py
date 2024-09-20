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

# model_type -> name for layers
MODELS_TO_LAYERS_MAPPING = {
    "bloom": "transformer.h",
    "gpt2": "transformer.h",
    "gpt_neox": "gpt_neox.layers",
    "llama": "model.layers",
    "opt": "model.decoder.layers",
}

# model_type -> name for embedding, these are the modules before the first layer
MODELS_TO_EMBEDDINGS_MAPPING = {
    "bloom": ["transformer.word_embeddings", "transformer.word_embeddings_layernorm"],
    "gpt2": ["transformer.wte", "transformer.wpe"],
    "gpt_neox": ["gpt_neox.embed_in"],
    "llama": ["model.embed_tokens"],
    "opt": [
        "model.decoder.embed_tokens",
        "model.decoder.embed_positions",
        "model.model.decoder.project_out",
        "model.model.decoder.project_in",
    ],
}

# model_type -> max length of the model, extracted from the config
# will be int if not present in the config
MODELS_TO_MAX_LENGTH_MAPPING = {
    "__default__": "max_position_embeddings",
    "bloom": 2048,
    "gpt2": "n_positions",
    "gpt_neox": "max_position_embeddings",
    "llama": "max_position_embeddings",
    "opt": "max_position_embeddings",
}


# To extend following list/map from huggingface config
# there is the priority order: NUM_HEADS_NAMES[0] and HIDDEN_SIZE_NAMES[0] are the first choice
# which means user can override the value in config file
NUM_HEADS_NAMES = (
    "num_heads",
    "num_attention_heads",
    "n_head",
    "n_heads",
    "encoder_attention_heads",
)
NUM_HIDDEN_LAYER_NAMES = ("num_hidden_layers", "num_layers", "n_layer", "n_layers")
NUM_KEY_VALUE_HEADS_NAMES = ("num_key_value_heads",)
HIDDEN_SIZE_NAMES = ("hidden_size", "dim", "d_model", "n_embd")
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

MODEL_OUTSIDE_LAYER_MODULES = {
    "phi3": ["model.embed_tokens", "embed_dropout", "model.norm"],
}

MODEL_INSIDE_LAYER_MODULES = {
    "phi3": [
        ["self_attn.qkv_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_up_proj"],
        ["mlp.down_proj"],
    ]
}

MODEL_LAYERS_BLOCK_NAME = {"phi3": "model.layers"}
