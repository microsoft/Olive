import torch.nn as nn


class MultipleInputsModel(nn.Module):
    def __init__(self):
        super(MultipleInputsModel, self).__init__()

    def forward(self, x, y, z):
        s = x + y + z
        return s


class MultipleOutputsModel(nn.Module):
    def __init__(self):
        super(MultipleOutputsModel, self).__init__()

    def forward(self, x):
        s = x + x
        return x, s


class MultipleInputAndOutputsModel(nn.Module):
    def __init__(self):
        super(MultipleInputAndOutputsModel, self).__init__()

    def forward(self, x, y):
        s = x + y
        m = x * y
        return s, m
