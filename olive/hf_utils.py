from typing import Callable

import torch


def load_huggingface_model_from_task(task: str, name: str):
    """Load huggingface model from task and name"""
    from transformers import AutoModel
    from transformers.pipelines import check_task

    task_results = check_task(task)
    assert isinstance(task_results, tuple)
    if len(task_results) == 2:
        targeted_task = task_results[0]
    elif len(task_results) == 3:
        targeted_task = task_results[1]
    else:
        raise ValueError("unsupported transfomers version")

    model_class = {"pt": targeted_task["pt"]}
    class_tuple = ()
    class_tuple = class_tuple + model_class.get("pt", (AutoModel,))

    model = None
    for model_class in class_tuple:
        try:
            model = model_class.from_pretrained(name)
            return model
        except (OSError, ValueError):
            continue

    return model


class ORTWhisperModel(torch.nn.Module):
    """ORT implementation of whisper model"""

    def __init__(self, encoder_decoder_init: torch.nn.Module, decoder: torch.nn.Module, config):
        super().__init__()
        self.encoder_decoder_init = encoder_decoder_init
        self.decoder = decoder
        self.config = config


def get_ort_whisper_for_conditional_generation(name: str):
    """Load ORT implementation of whisper model"""
    from onnxruntime.transformers.models.whisper.whisper_decoder import WhisperDecoder
    from onnxruntime.transformers.models.whisper.whisper_encoder_decoder_init import WhisperEncoderDecoderInit
    from transformers import WhisperForConditionalGeneration

    # get the original model
    model = WhisperForConditionalGeneration.from_pretrained(name)

    # ort implementation
    decoder = WhisperDecoder(model, None, model.config)
    encoder_decoder_init = WhisperEncoderDecoderInit(
        model,
        model,
        None,
        model.config,
        decoder_start_token_id=None,
    )

    return ORTWhisperModel(encoder_decoder_init, decoder, model.config)


MODEL_CLASS_TO_ORT_IMPLEMENTATION = {"WhisperForConditionalGeneration": get_ort_whisper_for_conditional_generation}


def huggingface_model_loader(model_loader):
    import transformers

    if model_loader is None:
        model_loader = "AutoModel"
    if isinstance(model_loader, str):
        try:
            model_loader = getattr(transformers, model_loader)
        except AttributeError:
            raise AttributeError(f"{model_loader} is not found in transformers")
    elif not isinstance(model_loader, Callable):
        raise ValueError("model_loader must be a callable or a string defined in transformers")

    return model_loader.from_pretrained


def load_huggingface_model_from_model_class(model_class: str, name: str, use_ort_implementation: bool = False):
    """
    Load huggingface model from model_loader and name

    If use_ort_implementation is True, then return ORT implementation of the model.
    """

    if not use_ort_implementation:
        huggingface_model_loader(model_class)(name)

    if model_class not in MODEL_CLASS_TO_ORT_IMPLEMENTATION:
        raise ValueError(f"There is no ORT implementation for {model_class}")
    return MODEL_CLASS_TO_ORT_IMPLEMENTATION[model_class](name)
