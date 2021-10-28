import os

from olive.conversion_config import ConversionConfig
from olive.convert import convert


def test_tf_frozen_pb():
    model_path = os.path.join(os.path.dirname(__file__), "models", "full_doran_frozen.pb")
    test_data_path = os.path.join(os.path.dirname(__file__), "models", "doran.npz")
    inputs_schema = [{"name": "title_lengths:0"}, {"name": "title_encoder:0"}, {"name": "ratings:0"}, {"name": "query_lengths:0"},
                     {"name": "passage_lengths:0"}, {"name": "features:0"}, {"name": "encoder:0"}, {"name": "decoder:0"}, {"name": "Placeholder:0"}]
    outputs_schema = [{"name": "output_identity:0"}, {"name": "loss_identity:0"}]
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_framework="tensorflow", sample_input_data_path=test_data_path)
    convert(cvt_config)


def test_tf_saved_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "saved_model")
    inputs_schema = [{"name": "X:0"}]
    outputs_schema = [{"name": "pred:0"}]
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_framework="tensorflow")
    convert(cvt_config)


def test_tf_checkpoint():
    model_path = os.path.join(os.path.dirname(__file__), "models", "tf_ckpt", "model.ckpt.meta")
    inputs_schema = [{"name": "input:0"}]
    outputs_schema = [{"name": "result:0"}]
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_framework="tensorflow")
    convert(cvt_config)


def test_pytorch():
    model_path = os.path.join(os.path.dirname(__file__), "models", "squeezenet1_1.pth")
    inputs_schema = [{"name": "input_0", "shape": [1, 3, 244, 244]}]
    outputs_schema = [{"name": "output_0"}]
    cvt_config = ConversionConfig(model_path=model_path, inputs_schema=inputs_schema, outputs_schema=outputs_schema,
                                  model_framework="pytorch")
    convert(cvt_config)