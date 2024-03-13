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
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_subprocess(cmd, env=None, cwd=None, check=False):
    logger.debug("Running command: %s", cmd)

    assert isinstance(cmd, (str, list)), f"cmd must be a string or a list, got {type(cmd)}."
    windows = platform.system() == "Windows"
    if isinstance(cmd, str):
        cmd = shlex.split(cmd, posix=not windows)

    try:
        out = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, check=check)
    except subprocess.CalledProcessError as e:
        err_msg = [
            f"Failed to run {cmd} with returncode {e.returncode}!",
            f"Stderr: {e.stderr.decode('utf-8')}",
            f"Stdout: {e.stdout.decode('utf-8')}",
            f"Env: {env}",
        ]
        logger.error("\n".join(err_msg))  # noqa: TRY400
        raise
    returncode = out.returncode
    stdout = out.stdout.decode("utf-8")
    stderr = out.stderr.decode("utf-8")

    return returncode, stdout, stderr


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
    with open(filename, "rb") as f:
        return hash_io_stream(f)


def hash_update_from_file(filename, hash_value):
    assert Path(filename).is_file()
    with open(str(filename), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_value.update(chunk)
    return hash_value


def hash_update_from_dir(directory, hash_value):
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash_value.update(path.name.encode())
        if path.is_file():
            hash_value = hash_update_from_file(path, hash_value)
        elif path.is_dir():
            hash_value = hash_update_from_dir(path, hash_value)
    return hash_value


def hash_dir(directory):
    return hash_update_from_dir(directory, hashlib.md5()).hexdigest()


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
        logger.warning("Could not get source code for %s. Hash will be based on name only.", function.__name__)
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
            logger.debug("Calling function '%s'. Try %d of %d...", func.__name__, num_tries + 1, max_tries)
            out = func(*args, **kwargs)
            logger.debug("Succeeded.")
            return out
        except exceptions:
            num_tries += 1
            if num_tries == max_tries:
                logger.exception("The operation failed after the maximum number of retries.")
                raise
            logger.debug("Failed. Retrying in %d seconds...", sleep_time)
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


def resolve_torch_dtype(dtype):
    """Get torch dtype from string or torch dtype.

    :param dtype: dtype to resolve. Can be a string (float16, torch.float16, etc) or torch dtype.
    :return: torch dtype.
    """
    import torch

    if isinstance(dtype, str):
        dtype = dtype.replace("torch.", "")
        try:
            dtype = getattr(torch, dtype)
        except AttributeError as e:
            raise AttributeError(f"Invalid dtype '{dtype}'.") from e
    assert isinstance(dtype, torch.dtype), f"dtype must be a string or torch.dtype, got {type(dtype)}."
    return dtype


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


def find_submodules(module, submodule_types, full_name=False):
    """Find all submodules of a given type in a module.

    :param module: module to search.
    :param submodule_type: type of submodule to search for. Can be a single type or a tuple of types.
    :param full_name: if True, return full name of submodule. Otherwise, return last part of submodule name.
    :return: list of submodules names.
    """
    submodules = set()
    for name, submodule in module.named_modules():
        if isinstance(submodule, submodule_types):
            if full_name:
                submodules.add(name)
            else:
                submodules.add(name.split(".")[-1])
    return list(submodules) if submodules else None


def huggingface_login(token: str):
    from huggingface_hub import login

    login(token=token)


def aml_runner_hf_login():
    import os

    hf_login = os.environ.get("HF_LOGIN")
    if hf_login:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        keyvault_name = os.environ.get("KEYVAULT_NAME")
        logger.debug("Getting token from keyvault %s", keyvault_name)

        credential = DefaultAzureCredential()
        secret_client = SecretClient(vault_url=f"https://{keyvault_name}.vault.azure.net/", credential=credential)
        token = secret_client.get_secret("hf-token").value
        huggingface_login(token)


def all_files(path, ignore=None):
    """Find all files in a directory recursively, optionally ignoring some paths.

    :param path: The path to the directory to search. Can be a string or a `Path` object.
    :param ignore: A callable similar to `ignore` parameter of `shutil.copytree`.
        E.g. `shutil.ignore_patterns('__pycache__')`.
    :return: A generator of `Path` objects.
    """
    for member in Path(path).iterdir():
        ignored = ignore(path, [member.name]) if ignore else set()
        if member.name in ignored:
            continue
        if member.is_dir():
            yield from all_files(member, ignore)
        else:
            yield member


def copy_dir(src_dir, dst_dir, ignore=None, **kwargs):
    """Copy a directory recursively using `shutil.copytree`.

    Handles shutil.Error exceptions that happen even though all files were copied successfully.

    :param src_dir: The source directory. Can be a string or a `Path` object.
    :param dst_dir: The destination directory. Can be a string or a `Path` object.
    :param ignore: A callable that is used as `ignore` parameter to `shutil.copytree`.
    :param kwargs: Additional kwargs to pass to `shutil.copytree`.
    """
    try:
        shutil.copytree(src_dir, dst_dir, ignore=ignore, **kwargs)
    except shutil.Error as e:
        src_dir = Path(src_dir).resolve()
        dst_dir = Path(dst_dir).resolve()
        # Check if all files were copied successfully
        # only check files names so it is not foolproof
        not_copied = [
            member.relative_to(src_dir)
            for member in all_files(src_dir, ignore)
            if not (dst_dir / member.relative_to(src_dir)).exists()
        ]
        if not_copied:
            raise RuntimeError(f"Failed to copy {not_copied}") from e
        else:
            logger.warning(
                "Received shutil.Error '%s' but all required file names exist in destination directory. "
                "Assuming all files were copied successfully and continuing.",
                e,
            )


def set_tempdir(tempdir: str = None):
    """Set the root directory for tempfiles.

    :param tempdir: new tempdir.
    """
    if tempdir is None:
        return

    tempdir = Path(tempdir).resolve()
    tempdir.mkdir(parents=True, exist_ok=True)
    # setting as string to be safe
    logger.debug("Setting tempdir to %s from %s", tempdir, tempfile.tempdir)
    tempfile.tempdir = str(tempdir)
