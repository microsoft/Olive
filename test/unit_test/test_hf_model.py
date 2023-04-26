import transformers

from olive.model import ModelStorageKind, PyTorchModel


def test_load_hf_model_by_name_and_type():
    model_name = "gpt2"
    hf_model_loader = "GPT2DoubleHeadsModel"
    task_type = "RocStories"

    model = PyTorchModel(
        model_storage_kind=ModelStorageKind.HuggingFaceModel,
        model_metadata={"hf_model_name": model_name, "hf_model_loader": hf_model_loader, "hf_task_type": task_type},
    )
    module = model.load_model()
    assert isinstance(module, transformers.GPT2DoubleHeadsModel)
