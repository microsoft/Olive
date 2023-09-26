# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import hashlib
import inspect
import io
import json
import logging
import pickle
import platform
import shlex
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)


def run_subprocess(cmd, env=None, cwd=None, check=False):  # pragma: no cover
    logger.debug(f"Running command: {cmd} with env: {env}")

    windows = platform.system() == "Windows"
    cmd = shlex.split(cmd, posix=not windows)
    if windows:
        path = env.get("PATH") if env else None
        cmd_exe = shutil.which(cmd[0], path=path)
        cmd[0] = cmd_exe
    out = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True)
    returncode = out.returncode
    stdout = out.stdout.decode("utf-8")
    stderr = out.stderr.decode("utf-8")
    if check and returncode != 0:
        raise RuntimeError(f"Command '{cmd}' failed with return code {returncode} and error: {stderr}")

    return returncode, stdout, stderr


def get_package_name_from_ep(execution_provider):
    provider_package_mapping = {
        "CPUExecutionProvider": ("onnxruntime", "ort-nightly"),
        "CUDAExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "TensorrtExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "RocmExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
        "OpenVINOExecutionProvider": ("onnxruntime-openvino", None),
        "DmlExecutionProvider": ("onnxruntime-directml", "ort-nightly-directml"),
    }
    return provider_package_mapping.get(execution_provider, ("onnxruntime", "ort-nightly"))


def hash_string(string):  # pragma: no cover
    md5_hash = hashlib.md5()
    md5_hash.update(string.encode())
    return md5_hash.hexdigest()


def hash_io_stream(f):  # pragma: no cover
    md5_hash = hashlib.md5()
    # Read and update hash in chunks of 4K
    for byte_block in iter(lambda: f.read(4096), b""):
        md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def hash_file(filename):  # pragma: no cover
    with open(filename, "rb") as f:  # noqa: PTH123
        return hash_io_stream(f)


def hash_dict(dictionary):  # pragma: no cover
    md5_hash = hashlib.md5()
    encoded_dictionary = json.dumps(dictionary, sort_keys=True).encode()
    md5_hash.update(encoded_dictionary)
    return md5_hash.hexdigest()


def hash_function(function):  # pragma: no cover
    md5_hash = hashlib.md5()
    try:
        source = inspect.getsource(function)
    except OSError:
        logger.warning(f"Could not get source code for {function.__name__}. Hash will be based on name only.")
        source = function.__name__
    md5_hash.update(source.encode())
    return md5_hash.hexdigest()


def hash_object(obj):  # pragma: no cover
    f = io.BytesIO()
    pickle.dump(obj, f)
    return hash_io_stream(f)


def unflatten_dict(dictionary):  # pragma: no cover
    """Unflatten a dictionary with keys of the form "a.b.c" into a nested dictionary."""
    result = {}
    for key, value in dictionary.items():
        parts = list(key)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def flatten_dict(dictionary, stop_condition=None):  # pragma: no cover
    """Flatten a nested dictionary into a dictionary with keys of the form (a,b,c)."""
    result = {}
    for key, value in dictionary.items():
        if stop_condition is not None and stop_condition(value):
            result[(key,)] = value
        elif isinstance(value, dict):
            result.update({(key, *k): v for k, v in flatten_dict(value, stop_condition).items()})
        else:
            result[(key,)] = value
    return result


def retry_func(func, args=None, kwargs=None, max_tries=3, delay=5, backoff=2, exceptions=None):
    """Retry a function call using an exponential backoff.

    Args:
        func: Function to call.
        args: Arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.
        max_tries: Maximum number of retries.
        delay: Initial delay between retries in seconds.
        backoff: Backoff multiplier e.g. value of 2 will double the delay each retry.
        exceptions: Exceptions to catch. If None, catch all exceptions. Can be a single exception or a tuple
            of exceptions.
    """
    args = args or []
    kwargs = kwargs or {}
    exceptions = exceptions or Exception
    num_tries, sleep_time = 0, delay
    while num_tries < max_tries:
        try:
            logger.debug(f"Calling function '{func.__name__}'. Try {num_tries + 1} of {max_tries}...")
            out = func(*args, **kwargs)
            logger.debug("Succeeded.")
            return out
        except exceptions as e:
            num_tries += 1
            if num_tries == max_tries:
                logger.error(f"Failed with error: {e}", exc_info=True)
                raise e
            logger.debug(f"Failed. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            sleep_time *= backoff
    return None


def tensor_data_to_device(data, device: str):
    if device is None:
        return data

    from torch import Tensor

    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: tensor_data_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensor_data_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(tensor_data_to_device(v, device) for v in data)
    elif isinstance(data, set):
        return {tensor_data_to_device(v, device) for v in data}
    else:
        return data


def get_attr(module, attr, fail_on_not_found=False):
    """Get attribute from module.

    :param module: module to get attribute from.
    :param attr: attribute name, can be a string with dot notation. If empty, return module.
    :param fail_on_not_found: if True, raise AttributeError if attribute is not found.
    :return: attribute
    """
    if not attr:
        # return module if attr is empty
        return module

    attr = attr.split(".")
    for a in attr:
        try:
            module = getattr(module, a)
        except AttributeError as e:
            not_found_message = f"Attribute {attr} not found."
            if fail_on_not_found:
                raise AttributeError(not_found_message) from e
            else:
                logger.warning(not_found_message)
                return None
    return module
