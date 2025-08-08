from test.utils import get_pytorch_model


def load_decoder_model(model_path):
    return get_pytorch_model().load_model()


def load_decoder_with_past_model(model_path):
    return get_pytorch_model().load_model()


def decoder_with_past_inputs(model):
    pass


def decoder_inputs(model):
    pass
