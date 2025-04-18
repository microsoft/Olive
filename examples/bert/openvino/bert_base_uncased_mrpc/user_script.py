# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import datasets
import numpy as np
import torch
from transformers import BertTokenizer

from olive.data.registry import Registry

# -------------------------------------------------------------------------
# Common Dataset
# -------------------------------------------------------------------------

seed = 0
# seed everything to 0 for reproducibility, https://pytorch.org/docs/stable/notes/randomness.html
# do not set random seed and np.random.seed for aml test, since it will cause aml job name conflict
torch.manual_seed(seed)
# the following are needed only for GPU
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set max sequence length the same as provided in the config JSON file
MAX_SEQ_LENGTH = 128

# define the tokenizer
tokenizer = BertTokenizer.from_pretrained("Intel/bert-base-uncased-mrpc")
VOCAB_SIZE = len(tokenizer)

# set default input
default_input = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.int64)

# define model inputs
model_inputs = {
    "input_ids": default_input,
    "attention_mask": default_input,
    "token_type_ids": default_input,
}

# capture input names
INPUT_NAMES = list(model_inputs)


@Registry.register_dataset()
def bert_base_uncased_mrpc_dataset():
    # load the raw wikipedia dataset for tuning. Load just 300 examples for speed.
    raw_dataset = datasets.load_dataset("glue", "mrpc", split="validation", trust_remote_code=True)

    def _preprocess_fn(examples):
        texts = (examples["sentence1"], examples["sentence2"])
        result = tokenizer(
            *texts,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )
        result["labels"] = examples["label"]
        return result

    # preprocess the datase
    return raw_dataset.map(_preprocess_fn, batched=True, batch_size=1)


def custom_transform_func(data_item):
    return {
        name: np.asarray([np.array([g.flatten() for g in data_item[name]]).flatten()], dtype=np.int64)
        for name in INPUT_NAMES
    }


def custom_example_func():
    vocab_size = VOCAB_SIZE
    batch_size = 1
    sequence_length = MAX_SEQ_LENGTH

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # Generate random attention_mask (1s for actual tokens, 0s for padding)
    attention_mask = default_input

    # Generate random token_type_ids (0 for sentence 1, 1 for sentence 2)
    token_type_ids = default_input

    return [input_ids, attention_mask, token_type_ids]
