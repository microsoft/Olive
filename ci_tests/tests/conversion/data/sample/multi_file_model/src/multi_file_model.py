import torch
import torch.onnx


class MultiFileModel(torch.nn.Module):

    def __init__(self):
        super(MultiFileModel, self).__init__()
        from .a.b.add import add_func
        from .sub import sub_func
        self.add_func = add_func
        self.sub_func = sub_func

    def forward(self, x):
        return self.add_func(x) + self.sub_func(x)


def save_model(path):
    # save model
    model = MultiFileModel()
    model.eval()
    torch.save(model, path)


def get_input_shapes():
    return [[2, 2]]


def get_input_names():
    return ["input_mf"]


def get_output_names():
    return ["output_mf"]
