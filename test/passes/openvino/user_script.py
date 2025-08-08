from test.utils import get_pytorch_model, get_pytorch_model_dummy_input


def get_dummy_input():
    input_model = get_pytorch_model()
    return get_pytorch_model_dummy_input(input_model)


def get_input():
    return [[1, 1]]
