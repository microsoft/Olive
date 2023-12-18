# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import collections
import collections.abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from olive.exception import OliveEvaluationError

logger = logging.getLogger(__name__)


def resolve_onnx_path(file_or_dir_path: str, model_filename: str = "model.onnx") -> str:
    """Get the model full path.

    The engine provides output paths to ONNX passes that do not contain .onnx extension
    (these paths are generally locations in the cache). This function will convert such
    paths to absolute file paths and also ensure the parent directories exist.
    If the input path is already an ONNX file it is simply returned. Examples:

    resolve_onnx_path("c:/foo/bar.onnx") -> c:/foo/bar.onnx

    resolve_onnx_path("c:/foo/bar") -> c:/foo/bar/model.onnx
    """
    if not model_filename.endswith(".onnx"):
        raise ValueError(f"ONNXModel's model name must end with '.onnx', got {model_filename}")

    path = Path(file_or_dir_path)
    if path.suffix != ".onnx":
        path = path / model_filename
        parent_dir = path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_onnx_file_path(model_path: str, onnx_file_name: Optional[str] = None) -> str:
    """Get the path to the ONNX model file.

    If model_path is a file, it is returned as is. If model_path is a
    directory, the onnx_file_name is appended to it and the resulting path is returned. If onnx_file_name is not
    specified, it is inferred if there is only one .onnx file in the directory, else an error is raised.
    """
    assert Path(model_path).exists(), f"Model path {model_path} does not exist"

    # if model_path is a file, return it as is
    if Path(model_path).is_file():
        return model_path

    # if model_path is a directory, append onnx_file_name to it
    if onnx_file_name:
        onnx_file_path = Path(model_path) / onnx_file_name
        assert onnx_file_path.exists(), f"ONNX model file {onnx_file_path} does not exist"
        return str(onnx_file_path)

    # try to infer onnx_file_name
    logger.warning(
        "model_path is a directory but onnx_file_name is not specified. Trying to infer it. It is recommended to"
        " specify onnx_file_name explicitly."
    )
    onnx_file_names = list(Path(model_path).glob("*.onnx"))
    if len(onnx_file_names) == 1:
        return str(onnx_file_names[0])
    elif len(onnx_file_names) > 1:
        raise ValueError(
            f"Multiple .onnx model files found in the model folder {model_path}. Please specify one using the"
            " onnx_file_name argument."
        )
    else:
        raise ValueError(f"No .onnx file found in the model folder {model_path}.")


def check_and_normalize_provider_args(
    providers: Sequence[Union[str, Tuple[str, Dict[Any, Any]]]],
    provider_options: Sequence[Dict[Any, Any]],
    available_provider_names: Sequence[str],
):
    """Validate the 'providers' and 'provider_options' arguments and returns a normalized version.

    :param providers: Optional sequence of providers in order of decreasing
        precedence. Values can either be provider names or tuples of
        (provider name, options dict).
    :param provider_options: Optional sequence of options dicts corresponding
        to the providers listed in 'providers'.
    :param available_provider_names: The available provider names.

    :return: Tuple of (normalized 'providers' sequence, normalized
        'provider_options' sequence).

    'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

    The normalized result is a tuple of:
    1. Sequence of provider names in the same order as 'providers'.
    2. Sequence of corresponding provider options dicts with string keys and
        values. Unspecified provider options yield empty dicts.
    """
    # This function is copied from the following file.
    #    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/onnxruntime_inference_collection.py
    if providers is None:
        return [], []

    provider_name_to_options = collections.OrderedDict()

    def set_provider_options(name, options):
        if name not in available_provider_names:
            logger.warning(
                "Specified provider '{}' is not in available provider names."
                "Available providers: '{}'".format(name, ", ".join(available_provider_names))
            )

        if name in provider_name_to_options:
            logger.warning(f"Duplicate provider '{name}' encountered, ignoring.")
            return

        normalized_options = {str(key): str(value) for key, value in options.items()}
        provider_name_to_options[name] = normalized_options

    if not isinstance(providers, collections.abc.Sequence):
        raise ValueError("'providers' should be a sequence.")

    if provider_options is not None:
        if not isinstance(provider_options, collections.abc.Sequence):
            raise ValueError("'provider_options' should be a sequence.")

        if len(providers) != len(provider_options):
            raise ValueError("'providers' and 'provider_options' should be the same length if both are given.")

        if not all(isinstance(provider, str) for provider in providers):
            raise ValueError("Only string values for 'providers' are supported if 'provider_options' is given.")

        if not all(isinstance(options_for_provider, dict) for options_for_provider in provider_options):
            raise ValueError("'provider_options' values must be dicts.")

        for name, options in zip(providers, provider_options):
            set_provider_options(name, options)

    else:
        for provider in providers:
            if isinstance(provider, str):
                set_provider_options(provider, {})
            elif (
                isinstance(provider, tuple)
                and len(provider) == 2
                and isinstance(provider[0], str)
                and isinstance(provider[1], dict)
            ):
                set_provider_options(provider[0], provider[1])
            else:
                raise ValueError("'providers' values must be either strings or (string, dict) tuples.")

    return list(provider_name_to_options.keys()), list(provider_name_to_options.values())


def check_ort_fallback(session, execution_providers):
    # pylint: disable=protected-access
    import onnxruntime as ort

    if execution_providers:
        if isinstance(execution_providers, tuple):
            assert len(execution_providers) == 2, "execution_providers must be a tuple of (provider, options)"
            execution_providers_to_check = [execution_providers]
        else:
            assert isinstance(execution_providers, (str, list))
            execution_providers_to_check = (
                [execution_providers] if isinstance(execution_providers, str) else execution_providers
            )
        execution_providers_to_check, _ = check_and_normalize_provider_args(
            execution_providers_to_check, None, ort.get_available_providers()
        )
        session_providers = session.get_providers()
        for ep in execution_providers_to_check:
            if ep not in session_providers:
                raise OliveEvaluationError(
                    f"The onnxruntime fallback happens. {ep} is not in the session providers {session_providers}."
                    f" session._enable_fallback = {session._enable_fallback}"
                )
        session.disable_fallback()
