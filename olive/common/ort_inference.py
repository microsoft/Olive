from pathlib import Path
from typing import Dict, Union


def get_ort_inference_session(
    model_path: Union[Path, str], inference_settings: Dict[str, any], use_ort_extensions: bool = False
):
    """Get an ONNXRuntime inference session."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    if use_ort_extensions:
        # register custom ops for onnxruntime-extensions
        from onnxruntime_extensions import get_library_path

        sess_options.register_custom_ops_library(get_library_path())

    # execution provider
    execution_provider = inference_settings.get("execution_provider")

    # session options
    session_options = inference_settings.get("session_options", {})
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
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        elif execution_mode == 1:
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    if graph_optimization_level:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel(graph_optimization_level)
    if extra_session_config:
        for key, value in extra_session_config.items():
            sess_options.add_session_config_entry(key, value)

    if isinstance(execution_provider, list):
        # execution_provider may be a list of tuples/lists where the first item in each tuple is the EP name
        execution_provider = [i[0] if isinstance(i, (tuple, list)) else i for i in execution_provider]
    elif isinstance(execution_provider, str):
        execution_provider = [execution_provider]

    for idx, ep in enumerate(execution_provider):
        if ep == "QNNExecutionProvider":
            # add backend_path for QNNExecutionProvider
            execution_provider[idx] = ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})
            break

    # dml specific settings
    if len(execution_provider) >= 1 and execution_provider[0] == "DmlExecutionProvider":
        sess_options.enable_mem_pattern = False

    provider_options = inference_settings.get("provider_options")

    # create session
    return ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=execution_provider, provider_options=provider_options
    )
