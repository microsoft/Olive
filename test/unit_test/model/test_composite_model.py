from test.unit_test.utils import get_onnx_model

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
