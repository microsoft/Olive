# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

import onnxruntime

# ruff: noqa


def run():
    # Load inference configuration json file
    with open("inference_config.json") as f:
        inference_settings = json.load(f)
    if inference_settings:
        session_options = inference_settings.get("session_options")
        execution_provider = inference_settings.get("execution_provider")
        use_ort_extensions = inference_settings.get("use_ort_extensions")

        # Onnxruntime inference session options
        sess_options = onnxruntime.SessionOptions()
        if use_ort_extensions:
            from onnxruntime_extensions import get_library_path

            sess_options.register_custom_ops_library(get_library_path())

        _update_sess_options(sess_options, session_options)
        session = onnxruntime.InferenceSession("model.onnx", sess_options, providers=execution_provider)
    else:
        # Use default inference session
        session = onnxruntime.InferenceSession("model.onnx")
    # Inference by session
    inputs = "input"  # input data
    output_names = "output"  # output names

    outputs = session.run([output_names], inputs)

    return outputs


def _update_sess_options(sess_options, session_options):
    inter_op_num_threads = session_options.get("inter_op_num_threads")
    intra_op_num_threads = session_options.get("intra_op_num_threads")
    execution_mode = session_options.get("execution_mode")
    graph_optimization_level = session_options.get("graph_optimization_level")
    extra_session_config = session_options.get("extra_session_config")
    if inter_op_num_threads:
        sess_options.inter_op_num_threads = inter_op_num_threads
    if intra_op_num_threads:
        sess_options.intra_op_num_threads = intra_op_num_threads
    if execution_mode:
        if execution_mode == 0:
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        elif execution_mode == 1:
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    if graph_optimization_level:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel(graph_optimization_level)
    if extra_session_config:
        for key, value in extra_session_config.items():
            sess_options.add_session_config_entry(key, value)
