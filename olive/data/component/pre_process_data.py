# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from copy import deepcopy
from typing import Any, Optional

from olive.common.hf.utils import get_model_config, get_tokenizer
from olive.data.component.dataset import BaseDataset, ClassificationDataset
from olive.data.component.text_generation import text_gen_pre_process
from olive.data.registry import Registry


@Registry.register_pre_process()
@Registry.register_default_pre_process()
@Registry.register_pre_process("skip_pre_process")
def pre_process(dataset, **kwargs):
    """Pre-process data.

    Args:
        dataset (object): Data to be pre-processed, reserved for internal dataset assignment.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.

    """
    return dataset


def _huggingface_pre_process_helper(dataset, map_func, max_samples, **kwargs):
    """Apply a map function to the dataset.

    Args:
        dataset (object): Data to be pre-processed.
        map_func (function): Function to be applied to the dataset.
        max_samples (int): Max number of samples to use.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.

    """
    if max_samples is not None:
        # select the data beforehand to avoid tokenizing the whole dataset
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    # output type is list
    tokenized_datasets = dataset.map(
        map_func,
        batched=kwargs.get("batched", True),
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format("torch", output_all_columns=True)
    return tokenized_datasets


def _create_tokenize_function(model_name, input_cols, label_col, trust_remote_code, kwargs):
    """Create a tokenization function with optional label processing.

    Args:
        model_name (str): Name of the huggingface model.
        input_cols (list): List of input columns.
        label_col (str, optional): Label column. If None, no label processing.
        trust_remote_code (bool, optional): Whether to trust remote code.
        kwargs (dict): Additional tokenization arguments.

    Returns:
        function: Tokenization function.

    """

    def _tokenize(examples):
        tokenizer = get_tokenizer(model_name, trust_remote_code=trust_remote_code)
        tokenized_inputs = tokenizer(
            *[examples[input_col] for input_col in input_cols if examples[input_col]],
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            max_length=kwargs.get("max_length"),
            is_split_into_words=kwargs.get("is_split_into_words", False),
            add_special_tokens=kwargs.get("add_special_tokens", True),
        )
        # Add label if label_col is provided
        if label_col is not None:
            tokenized_inputs["label"] = examples[label_col]
        return tokenized_inputs

    return _tokenize


@Registry.register_pre_process()
def huggingface_pre_process(
    dataset, model_name, input_cols, label_col="label", max_samples=None, trust_remote_code=None, **kwargs
):
    """Pre-process data.

    Args:
        dataset (object): Data to be pre-processed, reserved for internal dataset assignment.
        model_name (str): Name of the huggingface model.
        input_cols (list): List of input columns.
        label_col (str): Label column. Defaults to "label".
        max_samples (int, optional): Max number of samples to use. Defaults to None.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own
            modeling files. Defaults to None.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.

    """
    tokenize_func = _create_tokenize_function(model_name, input_cols, label_col, trust_remote_code, kwargs)

    model_config_path = kwargs.pop("model_config_path", None)
    # TODO(trajep): add the complete data operation mapping like:
    # align_labels -> align_labels_with_mapping
    # Also to support customized operation arguments from users
    if kwargs.pop("align_labels", False):
        model_hf_config = get_model_config(model_config_path or model_name, trust_remote_code=trust_remote_code)
        if model_hf_config and model_hf_config.label2id:
            dataset = dataset.align_labels_with_mapping(model_hf_config.label2id, label_col)

    tokenized_datasets = _huggingface_pre_process_helper(dataset, tokenize_func, max_samples, **kwargs)
    # label_col is "label" since we added label_col as "label" to tokenized_inputs
    return ClassificationDataset(tokenized_datasets, label_col="label", max_samples=max_samples)


@Registry.register_pre_process()
def tokenizer_pre_process(dataset, model_name, input_cols, max_samples=None, trust_remote_code=None, **kwargs):
    """Pre-process data for feature extraction task (no label processing).

    Args:
        dataset (object): Data to be pre-processed, reserved for internal dataset assignment.
        model_name (str): Name of the huggingface model.
        input_cols (list): List of input columns.
        max_samples (int, optional): Max number of samples to use. Defaults to None.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own
            modeling files. Defaults to None.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.

    """
    # Use the shared tokenization function with label_col=None
    tokenize_func = _create_tokenize_function(
        model_name, input_cols, label_col=None, trust_remote_code=trust_remote_code, kwargs=kwargs
    )

    tokenized_datasets = _huggingface_pre_process_helper(dataset, tokenize_func, max_samples, **kwargs)
    return BaseDataset(tokenized_datasets, max_samples=max_samples)


@Registry.register_pre_process()
def ner_huggingface_preprocess(
    dataset, model_name, input_cols, label_col="label", max_samples=None, trust_remote_code=None, **kwargs
):
    """Pre-process data for ner task."""

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
        tokenizer = get_tokenizer(model_name, trust_remote_code=trust_remote_code)
        tokenized_inputs = tokenizer(
            *[examples[input_col] for input_col in input_cols if examples[input_col]],
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            is_split_into_words=kwargs.get("is_split_into_words", True),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )
        all_labels = examples[label_col]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(_align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["label"] = new_labels
        return tokenized_inputs

    tokenized_datasets = _huggingface_pre_process_helper(dataset, _tokenizer_and_align_labels, max_samples, **kwargs)
    return ClassificationDataset(tokenized_datasets, label_col="label", max_samples=max_samples)


@Registry.register_pre_process()
def text_generation_huggingface_pre_process(
    dataset, model_name: str, trust_remote_code: Optional[bool] = None, **kwargs
):
    """Pre-process data for text generation task.

    Args:
        dataset (object): Data to be pre-processed, reserved for internal dataset assignment.
        model_name (str): Name of the huggingface model.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own
            modeling files. Defaults to None.
        **kwargs: Additional arguments.
            The common arguments are the fields in olive.data.component.text_generation.TextGenParams.

    """
    all_kwargs = deepcopy(kwargs)
    # task is not used in the pre-process function. Will pop it so that the config validation doesn't warn about
    # unused kwargs
    all_kwargs.pop("task", None)

    tokenizer = get_tokenizer(model_name, trust_remote_code=trust_remote_code)

    return text_gen_pre_process(dataset, tokenizer, all_kwargs)


@Registry.register_pre_process()
def audio_classification_pre_process(
    dataset,
    model_name: str,
    input_cols: list,
    label_col: str = "label",
    max_samples: Optional[int] = None,
    trust_remote_code: Optional[bool] = None,
    feature_extractor_args: Optional[dict[str, Any]] = None,
    **kwargs,
):
    """Pre-process data for audio classification task.

    Args:
        dataset (object): Data to be pre-processed, reserved for internal dataset assignment.
        model_name (str): Name of the huggingface model.
        input_cols (list): List of input columns.
        label_col (str): Label column. Defaults to "label".
        max_samples (int, optional): Max number of samples to use. Defaults to None.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own
            modeling files. Defaults to None.
        feature_extractor_args (dict, optional): Additional arguments for feature extractor.
        **kwargs: Additional arguments.
            The common arguments are the fields in olive.data.component.audio_classification.AudioClassificationParams.
            Extra arguments:
                - max_duration (int, optional): Max duration of audio in seconds. Defaults to 30.
                - labels_to_filter (list, optional): List of labels to filter. Defaults to None.
            Note: the AudioClassificationParams subclass already includes the common arguments.

    """
    from datasets import Audio
    from transformers import AutoFeatureExtractor

    assert len(input_cols) == 1, "Only one input column is supported for audio classification task."

    # align labels with model configs
    model_config = get_model_config(model_name, trust_remote_code=trust_remote_code)
    labels_to_filter = kwargs.get("labels_to_filter") or []
    dataset = dataset.filter(
        lambda x: x not in dataset.features["label"].str2int(labels_to_filter), input_columns=[label_col]
    )
    dataset = dataset.align_labels_with_mapping(model_config.label2id, label_col)

    fe_args = feature_extractor_args or {}
    fea_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=trust_remote_code, **fe_args)
    dataset.cast_column(input_cols[0], Audio(sampling_rate=fea_extractor.sampling_rate))

    def _tokenizer_and_align_labels(examples):
        max_duration = kwargs.get("max_duration", 30)

        audio_arrays = [x["array"] for x in examples[input_cols[0]]]
        tokenized_inputs = fea_extractor(
            audio_arrays,
            sampling_rate=fea_extractor.sampling_rate,
            max_length=int(fea_extractor.sampling_rate * max_duration),
            truncation=True,
            return_attention_mask=True,
        )

        tokenized_inputs["label"] = examples[label_col]

        return tokenized_inputs

    tokenized_datasets = _huggingface_pre_process_helper(dataset, _tokenizer_and_align_labels, max_samples, **kwargs)
    return ClassificationDataset(tokenized_datasets, label_col="label", max_samples=max_samples)


@Registry.register_pre_process()
def speech_transcription_pre_process(
    dataset,
    audio_col: str = "audio",
    text_col: str = "text",
    id_col: str = "",
    sample_rate: int = 16000,
    max_samples: Optional[int] = None,
    limit: Optional[float] = None,
    seed: int = 42,
    **kwargs,
):
    """Pre-process data for speech transcription (ASR) evaluation.

    Loads audio arrays and reference transcription text from a HuggingFace dataset.
    Returns a dataset of ({"audio": audio_array, "file_name": name}, reference_text) pairs
    suitable for WER evaluation.

    Args:
        dataset: HuggingFace dataset with audio and text columns.
        audio_col: Name of the audio column. Defaults to "audio".
        text_col: Name of the reference text column. Defaults to "text".
        id_col: Name of a column to use as a per-sample identifier (e.g. an audio file name or
            sample id). When set and present, its value is surfaced as ``file_name`` so it can be
            included in the evaluation sample log. Falls back to the HuggingFace Audio feature's
            ``path`` and then the dataset row index. Defaults to "".
        sample_rate: Target sample rate for audio. Defaults to 16000.
        max_samples: Maximum number of samples (deprecated, use limit). Defaults to None.
        limit: Sampling limit following Olive convention:
            If >= 1: use first N samples.
            If 0 < limit < 1: randomly sample that percentage.
            If 0 or None: use all samples.
        seed: Random seed for percentage-based sampling. Defaults to 42.
        **kwargs: Additional arguments.

    """
    from datasets import Audio

    dataset = dataset.cast_column(audio_col, Audio(sampling_rate=sample_rate))

    # Apply sampling: prefer limit over max_samples
    effective_limit = limit if limit is not None else (max_samples if max_samples else 0)
    if effective_limit and effective_limit != 0:
        from random import Random

        total = len(dataset)
        if 0 < effective_limit < 1:
            n = max(1, int(total * effective_limit))
            rng = Random(seed)
            indices = sorted(rng.sample(range(total), min(n, total)))
            dataset = dataset.select(indices)
        elif effective_limit >= 1:
            n = min(int(effective_limit), total)
            dataset = dataset.select(range(n))

    class SpeechTranscriptionDataset:
        """Dataset that returns ({"audio": audio_array, "file_name": name}, reference_text) pairs.

        Note: Use batch_size=1 in dataloader config as audio samples have variable lengths.
        """

        def __init__(self, hf_dataset, audio_column, text_column, id_column=""):
            self.dataset = hf_dataset
            self.audio_column = audio_column
            self.text_column = text_column
            self.id_column = id_column

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            import os

            import numpy as np

            item = self.dataset[idx]
            audio_item = item[self.audio_column]
            audio_array = np.array(audio_item["array"], dtype=np.float32)
            reference_text = item[self.text_column]

            path = audio_item.get("path") if isinstance(audio_item, dict) else None
            if self.id_column and self.id_column in item and item[self.id_column] is not None:
                file_name = str(item[self.id_column])
            elif path:
                file_name = os.path.basename(str(path))
            else:
                file_name = str(idx)

            return {"audio": audio_array, "file_name": file_name}, reference_text

        @staticmethod
        def collate_fn(batch):
            """Collate variable-length audio batches. Use with batch_size=1 or pad audio."""
            import numpy as np

            # batch_size=1 is expected for speech evaluation (variable-length audio)
            if len(batch) == 1:
                input_dict, text = batch[0]
                batched = {**input_dict, "audio": np.expand_dims(input_dict["audio"], 0)}
                return (batched, [text])
            # For batch_size > 1, return as lists (no padding)
            inputs = [item[0] for item in batch]
            texts = [item[1] for item in batch]
            return (inputs, texts)

    return SpeechTranscriptionDataset(dataset, audio_col, text_col, id_col)


@Registry.register_pre_process()
def vision_vqa_pre_process(
    dataset,
    image_col: str = "image",
    question_col: str = "question",
    answer_col: str = "answer",
    options_col: str = "",
    system_prompt: str = "",
    id_col: str = "",
    max_length: int = 4096,
    max_samples: Optional[int] = None,
    limit: Optional[float] = None,
    seed: int = 42,
    **kwargs,
):
    """Pre-process data for vision VQA evaluation.

    Loads image, question, and ground truth answer from a HuggingFace dataset.
    Returns a dataset of ({"image": image, "question": question}, answer) pairs.

    Note: This returns raw PIL images and question strings. For the PyTorch evaluator,
    the model's own processor/tokenizer should be applied within the model's forward
    method (or via a custom pre-process component). For the ONNX evaluator, provide a
    custom pre-process component that applies the appropriate processor/tokenizer to
    produce numeric tensors matching the model's io_config.

    Args:
        dataset: HuggingFace dataset with image, question, and answer columns.
        image_col: Name of the image column. Defaults to "image".
        question_col: Name of the question column. Defaults to "question".
        answer_col: Name of the answer column. Defaults to "answer".
        options_col: Name of the options column for multiple-choice questions. If specified,
            options are formatted as numbered choices and appended to the question. Defaults to "".
        system_prompt: System prompt to guide model responses (e.g., "Reply with only the
            option number"). Passed through to the evaluator. Defaults to "".
        id_col: Name of a column to use as a per-sample identifier (e.g. an image file name or
            sample id). When set and present, its value is surfaced as ``file_name`` so it can be
            included in the evaluation sample log. Falls back to the dataset row index. Defaults to "".
        max_length: Maximum generation length (input + output tokens) for the VLM. Vision prompts
            with large images can exceed 3000 tokens due to vision patches. Defaults to 4096.
        max_samples: Maximum number of samples (deprecated, use limit). Defaults to None.
        limit: Sampling limit following Olive convention:
            If >= 1: use first N samples.
            If 0 < limit < 1: randomly sample that percentage.
            If 0 or None: use all samples.
        seed: Random seed for percentage-based sampling. Defaults to 42.
        **kwargs: Additional arguments.

    """
    # Apply sampling: prefer limit over max_samples
    effective_limit = limit if limit is not None else (max_samples if max_samples else 0)
    if effective_limit and effective_limit != 0:
        from random import Random

        total = len(dataset)
        if 0 < effective_limit < 1:
            n = max(1, int(total * effective_limit))
            rng = Random(seed)
            indices = sorted(rng.sample(range(total), min(n, total)))
            dataset = dataset.select(indices)
        elif effective_limit >= 1:
            n = min(int(effective_limit), total)
            dataset = dataset.select(range(n))

    class VisionVQADataset:
        """Dataset that returns (input_dict, answer_text) pairs for VQA evaluation.

        Note: Use batch_size=1 in dataloader config as images have variable sizes.
        """

        def __init__(
            self,
            hf_dataset,
            image_column,
            question_column,
            answer_column,
            options_column="",
            sys_prompt="",
            id_column="",
            max_length=4096,
        ):
            self.dataset = hf_dataset
            self.image_column = image_column
            self.question_column = question_column
            self.answer_column = answer_column
            self.options_column = options_column
            self.system_prompt = sys_prompt
            self.id_column = id_column
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item[self.image_column]
            question = item[self.question_column]
            answer = item[self.answer_column]

            # Format options into the question if options_col is specified
            # Use 1-based numbering (1, 2, 3, 4) which aligns with how VLMs are
            # typically prompted and avoids confusion with diagram region labels.
            num_choices = 0
            if self.options_column and self.options_column in item:
                options = item[self.options_column]
                if isinstance(options, (list, tuple)) and len(options) > 0:
                    num_choices = len(options)
                    options_text = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))
                    question = f"{question}\n{options_text}"

            # Handle list/tuple answers (some datasets have multiple valid answers)
            # Join with | separator so metrics can match against any valid answer
            if isinstance(answer, (list, tuple)):
                answer = "|".join(str(a) for a in answer) if answer else ""

            # Convert 0-based answer index to 1-based to match the option numbering
            if num_choices > 0:
                try:
                    answer_idx = int(answer)
                    answer = str(answer_idx + 1)
                except (ValueError, TypeError):
                    pass  # answer is already a non-numeric string (e.g., text label)

            input_dict = {
                "image": image,
                "question": question,
                "system_prompt": self.system_prompt,
                "num_choices": num_choices,
                "max_length": self.max_length,
                "file_name": (
                    str(item[self.id_column])
                    if self.id_column and self.id_column in item and item[self.id_column] is not None
                    else str(idx)
                ),
            }
            return input_dict, str(answer)

        @staticmethod
        def collate_fn(batch):
            """Collate VQA batches. Use with batch_size=1 for variable-size images.

            Note: answers are always strings at this point (list/tuple answers are
            joined with "|" in __getitem__), so no list-of-lists issue arises.
            """
            if len(batch) == 1:
                input_dict, answer = batch[0]
                return (input_dict, [answer])
            inputs = [item[0] for item in batch]
            answers = [item[1] for item in batch]
            return (inputs, answers)

    return VisionVQADataset(
        dataset, image_col, question_col, answer_col, options_col, system_prompt, id_col, max_length
    )
