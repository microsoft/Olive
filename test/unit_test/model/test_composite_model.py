from pathlib import Path
from test.unit_test.utils import get_onnx_model

import torch

from olive.model.config.model_config import ModelConfig
from olive.model.handler.composite import CompositeModelHandler


def test_composite_model_to_json():
    input_models = []
    input_models.append(get_onnx_model())
    input_models.append(get_onnx_model())
    composite_model = CompositeModelHandler(
        input_models,
        ["encoder_decoder_init", "decoder"],
    )
    composite_json = composite_model.to_json()
    model_config = ModelConfig.from_json(composite_json)
    assert model_config.type == CompositeModelHandler.model_type


def test_composite_pytorch_model():
    current_dir = Path(__file__).parent.resolve()
    user_script_path = str(current_dir / "user_script.py")
    model_config = {
        "type": "CompositePyTorchModel",
        "config": {
            "model_components": [
                {
                    "name": "decoder_model",
                    "type": "PyTorchModel",
                    "config": {
                        "model_script": user_script_path,
                        "model_loader": "load_decoder_model",
                        "dummy_inputs_func": "decoder_inputs",
                    },
                },
                {
                    "name": "decoder_with_past_model",
                    "type": "PyTorchModel",
                    "config": {
                        "model_script": user_script_path,
                        "model_loader": "load_decoder_with_past_model",
                        "dummy_inputs_func": "decoder_with_past_inputs",
                    },
                },
            ]
        },
    }
    model_config = ModelConfig.from_json(model_config)
    model = model_config.create_model()

    assert model.model_type == "CompositePyTorchModel"
    for name, comp_model in model.get_model_components():
        assert name in ["decoder_model", "decoder_with_past_model"]
        assert comp_model.model_type == "PyTorchModel"
        assert comp_model.model_loader in ["load_decoder_model", "load_decoder_with_past_model"]
        assert comp_model.dummy_inputs_func in ["decoder_inputs", "decoder_with_past_inputs"]
        torch_nn_model = comp_model.load_model()
        assert isinstance(torch_nn_model, torch.nn.Module)
