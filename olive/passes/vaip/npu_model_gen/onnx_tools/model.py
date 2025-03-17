##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import os
import pathlib

import onnx

from .graph import Graph
from .utils import ModelConfig


class Model:
    def __init__(self, m: [str, onnx.ModelProto, pathlib.Path], mcfg={}):
        self.modelname = ""
        self.cfg = ModelConfig(mcfg)
        if isinstance(m, pathlib.Path):
            self.modelname = m.stem
            m = onnx.load_model(m)
        elif isinstance(m, str):
            self.modelname = os.path.basename(m)
            self.modelname = os.path.splitext(self.modelname)[0]
            m = onnx.load_model(m)
        if not isinstance(m, onnx.ModelProto):
            self.valid = False
            return
        self.valid = True
        self.mproto = m
        self.graph = Graph(m.graph, self.cfg)

    def save_model(self, f: str, shape_only: bool = False, no_shape: bool = False):
        self.graph.save_model(
            f, shape_only=shape_only, rawmodel=self.mproto, no_shape=no_shape
        )
