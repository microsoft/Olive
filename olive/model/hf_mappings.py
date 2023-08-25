# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# mapping from task to feature
TASK_TO_FEATURE = {
    "automatic-speech-recognition": "speech2seq-lm",
    "fill-mask": "masked-lm",
    "image-classification": "image-classification",
    "image-segmentation": "image-segmentation",
    "image-to-text": "vision2seq-lm",
    "multiple-choice": "multiple-choice",
    "ner": "token-classification",
    "object-detection": "object-detection",
    "question-answering": "question-answering",
    "sentiment-analysis": "sequence-classification",
    "summarization": "seq2seq-lm",
    "text2text-generation": "seq2seq-lm",
    "text-classification": "sequence-classification",
    "text-generation": "causal-lm",
    "token-classification": "token-classification",
    "translation": "seq2seq-lm",
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
