# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


@Registry.register_default_pre_process()
def pre_process(_dataset):
    """Pre-process data.

    Args:
        data (object): Data to be pre-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    return _dataset


def _huggingface_pre_process_helper(dataset, model_name, input_cols, label_cols, map_func, **kwargs):
    """Pre-process data.

    Args:
        data (object): Data to be pre-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    # output type is list
    tokenized_datasets = dataset.map(
        map_func,
        batched=kwargs.get("batched", True),
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format("torch", output_all_columns=True)
    return tokenized_datasets


@Registry.register_pre_process()
def huggingface_pre_process(_dataset, model_name, input_cols, label_cols, max_samples=None, **kwargs):
    """Pre-process data.

    Args:
        data (object): Data to be pre-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    from transformers import AutoTokenizer

    def _tokenizer_and_align_labels(examples):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_inputs = tokenizer(
            *[examples[input_col] for input_col in input_cols],
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            is_split_into_words=kwargs.get("is_split_into_words", False),
            add_special_tokens=kwargs.get("add_special_tokens", True),
        )
        # TODO: support multiple label columns if needed
        tokenized_inputs["label"] = examples[label_cols[0]]
        # huggingface dataset api limit to return dict and arrow table
        return tokenized_inputs

    tokenized_datasets = _huggingface_pre_process_helper(
        _dataset, model_name, input_cols, label_cols, _tokenizer_and_align_labels, **kwargs
    )
    # label_cols is ["label"] since we added label_cols[0] as "label" to tokenized_inputs
    return BaseDataset(tokenized_datasets, label_cols=["label"], max_samples=max_samples)


@Registry.register_pre_process()
def ner_huggingface_preprocess(_dataset, model_name, input_cols, label_cols, max_samples=None, **kwargs):
    """
    Pre-process data for ner task.
    """
    from transformers import AutoTokenizer

    def _align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = 0 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(0)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        return new_labels

    def _tokenizer_and_align_labels(examples):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_inputs = tokenizer(
            *[examples[input_col] for input_col in input_cols],
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            is_split_into_words=kwargs.get("is_split_into_words", True),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )
        all_labels = examples[label_cols[0]]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(_align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["label"] = new_labels
        return tokenized_inputs

    tokenized_datasets = _huggingface_pre_process_helper(
        _dataset, model_name, input_cols, label_cols, _tokenizer_and_align_labels, **kwargs
    )
    return BaseDataset(tokenized_datasets, label_cols=["label"], max_samples=max_samples)


@Registry.register_pre_process()
def text_generation_huggingface_pre_process(
    _dataset, model_name, input_cols, seqlen, stride=None, max_samples=None, **kwargs
):
    """Pre-process data.

    Args:
        _dataset (object): Data to be pre-processed.
        model_name (str): Name of the huggingface model.
        input_cols (list): List of input columns.
        seqlen (int): Length of the sequence.
        stride (int): Stride to use when splitting the sequence. I
            If None, the sequence is split into non-overlapping sequences. No context is used.
            When stride is not None, we use a sliding window approach to split the sequence. The stride is
            also used as context length.
        max_samples (int): Maximum number of samples to use.
        **kwargs: Additional arguments.
            joiner (str): Delimiter to use when joining the rows of the input columns.
            random_seed (int): Random seed to use. If not None, we use the random seed to choose the starting
                point of the sequence.

    Returns:
        object: Pre-processed data.
    """
    from random import Random

    from datasets import Dataset as HFDataset
    from transformers import AutoTokenizer

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # gather text from all input columns
    text_list = []
    for input_col in input_cols:
        text_list += _dataset[input_col]
    # delimiter between the text sequences
    joiner = kwargs.get("joiner", " ")
    text = joiner.join(text_list)

    # in order to make processing faster we will only tokenize as much as needed
    # assumes that num words > num tokens
    split_text = text.split(" ")
    num_text = len(split_text)

    tokenized_inputs = {
        "input_ids": [],
        "target_ids": [],
    }

    random_seed = kwargs.get("random_seed")
    if random_seed is None:
        # no randomization, just use contiguous blocks of tokens
        max_text = max_samples * seqlen if max_samples is not None else num_text
        max_text = min(max_text, num_text)
        encodings = tokenizer(" ".join(split_text[:max_text]), return_tensors="pt")
        input_ids = encodings.input_ids

        num_tokens = input_ids.shape[1]
        step = stride or seqlen
        # loop over the number of tokens
        # all inputs must be seqlen long
        for begin_loc in range(0, num_tokens - seqlen, step):
            # end_loc is the beginning of the next sequence
            end_loc = begin_loc + seqlen
            # get the input sequence
            input_ids = encodings.input_ids[0, begin_loc:end_loc].clone()
            # target is the same as input, but shifted one token over
            target_ids = encodings.input_ids[0, begin_loc + 1 : end_loc + 1].clone()  # noqa: E203

            # set to -100 to ignore loss for context
            if stride is not None:
                target_ids[:-stride] = -100

            # add to list
            tokenized_inputs["input_ids"].append(input_ids)
            tokenized_inputs["target_ids"].append(target_ids)
    else:
        assert max_samples, "max_samples must be specified if random_seed is None"
        # randomization, sample random blocks of tokens
        rng = Random(random_seed)

        for _ in range(max_samples):
            # -2 since we need to leave space for the target
            begin_loc = rng.randint(0, num_text - seqlen - 2)
            encodings = tokenizer(
                " ".join(split_text[begin_loc : begin_loc + seqlen + 2]), return_tensors="pt"  # noqa: E203
            )

            if encodings.input_ids.shape[1] < seqlen + 1:
                # in case the encoding is too short
                continue

            input_ids = encodings.input_ids[0, :seqlen].clone()
            target_ids = encodings.input_ids[0, 1 : seqlen + 1].clone()  # noqa: E203

            # set to -100 to ignore loss for context
            if stride is not None:
                target_ids[:-stride] = -100

            # add to list
            tokenized_inputs["input_ids"].append(input_ids)
            tokenized_inputs["target_ids"].append(target_ids)

    # convert to HFDataset
    hf_dataset = HFDataset.from_dict(tokenized_inputs)
    hf_dataset.set_format("torch", output_all_columns=True)

    # return BaseDataset
    return BaseDataset(hf_dataset, ["target_ids"], max_samples=max_samples)
