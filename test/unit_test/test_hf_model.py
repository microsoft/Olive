import torch

from olive.model import ModelStorageKind, PyTorchModel


def test_load_hf_model_by_name_and_type():
    # The model name and task type is gotten from
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines#transformers.pipeline
    model_name = "facebook/wav2vec2-base-960h"
    task_type = "automatic-speech-recognition"

    model = PyTorchModel(
        model_storage_kind=ModelStorageKind.HuggingFaceModel,
        model_metadata={"hf_model_name": model_name, "hf_task_type": task_type},
    )
    module = model.load_model()
    assert isinstance(module, torch.nn.Module)
