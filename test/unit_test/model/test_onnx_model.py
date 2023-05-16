# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.onnx_mlflow import save_model as mlflow_save_model
import tempfile
from pathlib import Path
from test.unit_test.utils import get_onnx_model
import mlflow
import onnxruntime as ort


def test_model_save_load_onnx_mlflow_format():
    input_model = get_onnx_model()
    model_proto = input_model.load_model()

    onnx_session_options = {
        "execution_mode": "sequential",
        "graph_optimization_level": 99, 
        "intra_op_num_threads": 19
        }

    with tempfile.TemporaryDirectory() as tempdir:
        output_dir = str(Path(tempdir) / "onnx_mlflow")

        mlflow_save_model(model_proto, output_dir, onnx_session_options=onnx_session_options)

        # Loading pyfunc model
        pyfunc_loaded = mlflow.pyfunc.load_model(output_dir)
        session_options = pyfunc_loaded._model_impl.rt.get_session_options()

        assert session_options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
        assert session_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        assert session_options.intra_op_num_threads == 19
