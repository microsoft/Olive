# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import mlflow.tracking
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from packaging.version import Version

import olive

FLAVOR_NAME = "olive_onnx"
ONNX_EXECUTION_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def save_model(
    onnx_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    onnx_execution_providers=None,
    onnx_session_options=None,
    metadata=None,
):

    import onnx

    if onnx_execution_providers is None:
        onnx_execution_providers = ONNX_EXECUTION_PROVIDERS

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata
    model_data_subpath = "model.onnx"
    model_data_path = os.path.join(path, model_data_subpath)

    # Save onnx-model
    if Version(onnx.__version__) >= Version("1.9.0"):
        onnx.save_model(onnx_model, model_data_path, save_as_external_data=True)
    else:
        onnx.save_model(onnx_model, model_data_path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="olive.common.onnx_mlflow",
        data=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )

    _validate_onnx_session_options(onnx_session_options)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        onnx_version=onnx.__version__,
        data=model_data_subpath,
        providers=onnx_execution_providers,
        onnx_session_options=onnx_session_options,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def get_default_pip_requirements():
    return list(
        map(
            _get_pinned_requirement,
            [
                "onnx",
                # The ONNX pyfunc representation requires the OnnxRuntime
                # inference engine. Therefore, the conda environment must
                # include OnnxRuntime
                "onnxruntime",
                "olive-ai",
            ],
        )
    )


def get_default_conda_env():
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _load_pyfunc(path):
    return _OliveOnnxModelWrapper(path)


class _OliveOnnxModelWrapper:
    def __init__(self, path, providers=None):

        local_path = str(Path(path).parent)
        model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

        # Check if the MLModel config has the providers meta data
        if "providers" in model_meta.flavors.get(FLAVOR_NAME).keys():
            providers = model_meta.flavors.get(FLAVOR_NAME)["providers"]
        # If not, then default to the predefined list.
        else:
            providers = ONNX_EXECUTION_PROVIDERS

        sess_options = ort.SessionOptions()
        options = model_meta.flavors.get(FLAVOR_NAME)["onnx_session_options"]
        if options:
            inter_op_num_threads = options.get("inter_op_num_threads")
            intra_op_num_threads = options.get("intra_op_num_threads")
            execution_mode = options.get("execution_mode")
            graph_optimization_level = options.get("graph_optimization_level")
            extra_session_config = options.get("extra_session_config")
            if inter_op_num_threads:
                sess_options.inter_op_num_threads = inter_op_num_threads
            if intra_op_num_threads:
                sess_options.intra_op_num_threads = intra_op_num_threads
            if execution_mode:
                if execution_mode.upper() == "SEQUENTIAL":
                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                elif execution_mode.upper() == "PARALLEL":
                    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            if graph_optimization_level:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel(graph_optimization_level)
            if extra_session_config:
                for key, value in extra_session_config.items():
                    sess_options.add_session_config_entry(key, value)

        self.rt = ort.InferenceSession(path, providers=providers, sess_options=sess_options)

        assert len(self.rt.get_inputs()) >= 1
        self.inputs = [(inp.name, inp.type) for inp in self.rt.get_inputs()]
        self.output_names = [outp.name for outp in self.rt.get_outputs()]

    def _cast_float64_to_float32(self, feeds):
        for input_name, input_type in self.inputs:
            if input_type == "tensor(float)":
                feed = feeds.get(input_name)
                if feed is not None and feed.dtype == np.float64:
                    feeds[input_name] = feed.astype(np.float32)
        return feeds

    def predict(self, data):
        if isinstance(data, dict):
            feed_dict = data
        elif isinstance(data, np.ndarray):
            # NB: We do allow scoring with a single tensor (ndarray) in order to be compatible with
            # supported pyfunc inputs iff the model has a single input. The passed tensor is
            # assumed to be the first input.
            if len(self.inputs) != 1:
                inputs = [x[0] for x in self.inputs]
                raise MlflowException(
                    "Unable to map numpy array input to the expected model "
                    "input. "
                    "Numpy arrays can only be used as input for MLflow ONNX "
                    "models that have a single input. This model requires "
                    "{} inputs. Please pass in data as either a "
                    "dictionary or a DataFrame with the following tensors"
                    ": {}.".format(len(self.inputs), inputs)
                )
            feed_dict = {self.inputs[0][0]: data}
        elif isinstance(data, pd.DataFrame):
            if len(self.inputs) > 1:
                feed_dict = {name: data[name].values for (name, _) in self.inputs}
            else:
                feed_dict = {self.inputs[0][0]: data.values}

        else:
            raise TypeError(
                "Input should be a dictionary or a numpy array or a pandas.DataFrame, " "got '{}'".format(type(data))
            )

        feed_dict = self._cast_float64_to_float32(feed_dict)
        predicted = self.rt.run(self.output_names, feed_dict)

        if isinstance(data, pd.DataFrame):

            def format_output(data):
                data = np.asarray(data)
                return data.reshape(-1)

            response = pd.DataFrame.from_dict({c: format_output(p) for (c, p) in zip(self.output_names, predicted)})
            return response
        else:
            return dict(zip(self.output_names, predicted))


def load_model(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return _load_model(model_file=onnx_model_artifacts_path)


def log_model(
    onnx_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    onnx_execution_providers=None,
    onnx_session_options=None,
    metadata=None,
):
    return Model.log(
        artifact_path=artifact_path,
        flavor=olive.common.onnx_mlflow,
        onnx_model=onnx_model,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        onnx_execution_providers=onnx_execution_providers,
        onnx_session_options=onnx_session_options,
        metadata=metadata,
    )


def _load_model(model_file):
    import onnx

    onnx.checker.check_model(model_file)
    onnx_model = onnx.load(model_file)
    # Check Formation
    return onnx_model


def _validate_onnx_session_options(onnx_session_options):
    """
    Validates that the specified onnx_session_options dict is valid.
    :param ort_session_options: The onnx_session_options dict to validate.
    """
    if onnx_session_options is not None:
        if not isinstance(onnx_session_options, dict):
            raise TypeError("Argument onnx_session_options should be a dict, not {}".format(type(onnx_session_options)))
        for key, value in onnx_session_options.items():
            if key != "extra_session_config" and not hasattr(ort.SessionOptions, key):
                raise ValueError(
                    f"Key {key} in onnx_session_options is not a valid " "ONNX Runtime session options key"
                )
            elif key == "extra_session_config" and value and not isinstance(value, dict):
                raise TypeError(f"Value for key {key} in onnx_session_options should be a dict, " "not {type(value)}")
            elif key == "execution_mode" and value.upper() not in ["PARALLEL", "SEQUENTIAL"]:
                raise ValueError(
                    f"Value for key {key} in onnx_session_options should be " "'parallel' or 'sequential', not {value}"
                )
            elif key == "graph_optimization_level" and value not in [0, 1, 2, 99]:
                raise ValueError(
                    f"Value for key {key} in onnx_session_options should be 0, 1, 2, or 99, " "not {value}"
                )
            elif key in ["intra_op_num_threads", "intra_op_num_threads"] and value < 0:
                raise ValueError(f"Value for key {key} in onnx_session_options should be >= 0, not {value}")
