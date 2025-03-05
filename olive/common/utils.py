# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import codecs
import gc
import hashlib
import inspect
import io
import json
import logging
import os
import pickle
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


if sys.version_info >= (3, 11):
    from enum import IntEnum, StrEnum

    class StrEnumBase(StrEnum):
        pass

    class IntEnumBase(IntEnum):
        pass

else:
    from enum import Enum

    class StrEnumBase(str, Enum):
        def __str__(self) -> str:
            return self.value

    class IntEnumBase(int, Enum):
        pass


def run_subprocess(cmd, env=None, cwd=None, check=False):
    logger.debug("Running command: %s", cmd)

    assert isinstance(cmd, (str, list)), f"cmd must be a string or a list, got {type(cmd)}."
    windows = platform.system() == "Windows"
    if isinstance(cmd, str):
        # In posix model, the cmd string will be handled with specific posix rules.
        # https://docs.python.org/3.8/library/shlex.html#parsing-rules
        # We listed 2 typical examples:
        # 1. The cmd may contain the folder/file path from windows. This kind of path is like: C:\\User\\xxx\\...
        #   in posix mode, the shlex.split will split the path into ['C:Userxxx...'], which is not correct.
        #   in non-posix mode, the shlex.split will split the path into ['C:\\User\\xxx\\...'].
        # 2. The cmd may contain the quotes("" or ''). This kind of cmd is like: "str_0, 'str_1'"
        #   in posix mode, the shlex.split will split the cmd into ["str_0", "str_1"]
        #   in non-posix mode, the shlex.split will split the cmd into ["str_0", "'str_1'"]
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
    md5_hash = hashlib.sha256()
    md5_hash.update(string.encode())
    return md5_hash.hexdigest()


def hash_io_stream(f, block_size=4096):  # pragma: no cover
    md5_hash = hashlib.sha256()
    # Read and update hash in chunks of 4K
    for byte_block in iter(lambda: f.read(block_size), b""):
        md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def hash_file(filename, block_size=4096):  # pragma: no cover
    with open(filename, "rb") as f:
        return hash_io_stream(f, block_size)


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
    return hash_update_from_dir(directory, hashlib.sha256()).hexdigest()


def hash_dict(dictionary):  # pragma: no cover
    md5_hash = hashlib.sha256()
    encoded_dictionary = json.dumps(dictionary, sort_keys=True).encode()
    md5_hash.update(encoded_dictionary)
    return md5_hash.hexdigest()


def hash_function(function):  # pragma: no cover
    md5_hash = hashlib.sha256()
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


def get_nested_dict_value(dictionary: dict, key: Union[str, Tuple, List[str]]):
    """Get value from a nested dictionary."""
    if isinstance(key, str):
        key = [key]

    for k in key:
        dictionary = dictionary[k]
    return dictionary


def set_nested_dict_value(dictionary: dict, key: Union[str, Tuple, List[str]], new_value):
    """Replace value in a nested dictionary."""
    if isinstance(key, str):
        key = [key]

    for k in key[:-1]:
        dictionary = dictionary[k]
    dictionary[key[-1]] = new_value


def dict_diff(dict1: Optional[dict], dict2: Optional[dict]) -> Optional[dict]:
    """Return all members of dict1 that are not in dict2 or have different values."""
    dict1 = dict1 or {}
    dict2 = dict2 or {}
    return {k: v for k, v in dict1.items() if k not in dict2 or dict2[k] != v} or None


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
        except exceptions as e:
            num_tries += 1
            if num_tries == max_tries:
                logger.exception("The operation failed after the maximum number of retries.")
                raise
            logger.debug("Failed with %s. Retrying in %d seconds...", e, sleep_time)
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


def tensor_data_to_dtype(data, dtype):
    import torch

    if dtype is None:
        return data

    from torch import Tensor

    if isinstance(data, Tensor) and data.dtype in {torch.bfloat16, torch.float16, torch.float32, torch.float64}:
        return data.to(dtype)
    if isinstance(data, dict):
        return {k: tensor_data_to_dtype(v, dtype) for k, v in data.items()}
    if isinstance(data, list):
        return [tensor_data_to_dtype(v, dtype) for v in data]
    if isinstance(data, tuple):
        return tuple(tensor_data_to_dtype(v, dtype) for v in data)
    if isinstance(data, set):
        return {tensor_data_to_dtype(v, dtype) for v in data}
    return data


def format_data(data, io_config):
    """Format data based on io_config.

    :param data: data to format. Consists of torch tensors or numpy arrays.
        Single tensor or list of tensors: zipped with input names.
        Dict: Keys not in input names are ignored. So unused data is allowed.
        Caller needs to ensure the required inputs are present in the data.
    :param io_config: io config to use for formatting.
        input_names: list of input names.
        input_types: list of numpy input types.
    :return: formatted data. Consists of numpy arrays.
    """
    import numpy as np
    import torch

    input_names = io_config["input_names"]
    name_to_type = dict(zip(io_config["input_names"], io_config["input_types"]))
    if isinstance(data, list):
        # the input is just a list of tensors
        data = dict(zip(input_names, data))
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        # input is a single tensor
        data = dict(zip(input_names, [data]))
    elif not isinstance(data, dict):
        raise ValueError(f"Invalid input data format: {data}")
    return {
        k: np.ascontiguousarray(
            data[k].cpu().numpy() if isinstance(data[k], torch.Tensor) else data[k],
            dtype=name_to_type[k],
        )
        for k in data
        if k in input_names
    }


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
                logger.debug(not_found_message)
                return None
    return module


def set_attr(module, attr, submodule, fail_on_not_found=False):
    """Set attribute from module.

    :param module: module to set
    :param attr: attribute name, can be a string with dot notation.
    :param submodule: submodule to set.
    :param fail_on_not_found: if True, raise AttributeError if attribute is not found.
    """
    parent_name = ".".join(attr.split(".")[:-1])
    parent_module = get_attr(module, parent_name, fail_on_not_found)
    target_name = attr.split(".")[-1]

    setattr(parent_module, target_name, submodule)


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


def replace_submodules(module, submodule_types, new_submodule_func):
    """Replace all submodules of a given type in a module.

    :param module: module to search.
    :param submodule_type: type of submodule to search for. Can be a single type or a tuple of types.
    :param new_submodule_func: function to create new submodule. Should take old submodule as input.
    """
    for name, submodule in module.named_modules():
        if isinstance(submodule, submodule_types):
            set_attr(module, name, new_submodule_func(submodule))


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


def hardlink_copy_file(src, dst, *, follow_symlinks=True):
    """Copy a file using hardlink if possible, otherwise use shutil.copy2.

    Similar to shutil.copy2, if the destination is a directory, the file will be copied into the directory with the
    same name.
    If the destination file exists, it will be overwritten.
    """
    src = Path(src).resolve()  # NOTE: This call resolves any symlinks
    dst = Path(dst).resolve()

    if not src.exists():
        raise ValueError("Input source doesn't exist.", src)
    elif not src.is_file():
        raise ValueError("Input source is expected to be a file.", src)

    if dst.is_dir():
        dst = Path(dst) / src.name

    if dst.exists():
        logger.debug("Destination %s already exists. Removing.", dst)
        dst.unlink()

    try:
        os.link(src, dst, follow_symlinks=follow_symlinks)
    except OSError as e:
        # for instance, hardlinking across filesystems is not supported
        logger.debug("Linking failed with %s. Copying.", e)
        shutil.copy2(src, dst, follow_symlinks=follow_symlinks)


def hardlink_copy_dir(src_dir, dst_dir, **kwargs):
    """Copy a directory recursively using hardlinks. If hardlinking is not possible, use shutil.copy2.

    All kwargs are the same as shutil.copytree except for copy_function which is not supported.
    """
    # TODO(olivedevteam): What if dst_dir is a sub-directory of src_dir?
    # i.e. dst_dir.is_relative_to(src_dir) == True
    # Unsure if even shutil.copytree handles this scenario gracefully!!

    if kwargs.pop("copy_function", None) is not None:
        logger.warning("copy_function is not supported for hardlink_copy_dir. Ignoring.")

    if kwargs.pop("dirs_exist_ok", None) is not None:
        logger.warning("dirs_exist_ok is not supported for hardlink_copy_dir. Ignoring.")

    copy_dir(src_dir, dst_dir, copy_function=hardlink_copy_file, dirs_exist_ok=True, **kwargs)


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


def exclude_keys(original_dict: Dict, keys_to_exclude):
    return {k: v for k, v in original_dict.items() if k not in keys_to_exclude}


def find_first_matched_value(original, keys: Union[str, Tuple, List[str]], raise_key_error=False):
    if isinstance(keys, str):
        keys = [keys]

    for possible_name in keys:
        if isinstance(original, dict) and possible_name in original:
            return original[possible_name]
        elif hasattr(original, possible_name):
            return getattr(original, possible_name)

    if raise_key_error:
        raise KeyError(f"Keys {keys} not found in {original}")
    return None


def get_credentials(default_auth_params: Dict = None):
    """Get credentials for MLClient.

    Order of credential providers:
    1. Azure CLI
    2. DefaultAzureCredential
    3. InteractiveBrowserCredential
    """
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

    logger.debug("Getting credentials for MLClient")
    try:
        default_auth_params = default_auth_params or {}
        credential = DefaultAzureCredential(**default_auth_params)
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
        logger.debug("Using DefaultAzureCredential")
    except Exception:
        logger.warning("Using InteractiveBrowserCredential since of default credential errors", exc_info=True)
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    return credential


def hf_repo_exists(repo_name: str):
    try:
        from huggingface_hub import repo_exists
    except ImportError:
        logger.exception(
            "huggingface_hub is not installed. Please install huggingface_hub to support Huggingface model."
        )
        raise

    return repo_exists(repo_name)


class WeightsFileFormat(StrEnumBase):
    PT = "pt"
    NUMPY = "numpy"
    SAFETENSORS = "safetensors"
    ONNX_ADAPTER = "onnx_adapter"


def save_weights(weights: Dict, path: Union[str, Path], file_format: WeightsFileFormat = WeightsFileFormat.NUMPY):
    """Save the weights to a file.

    :param weights: Dictionary of numpy arrays.
    :param path: Path to save the weights. Might or might not include the file extension.
    :param file_format: Format to save the weights in.
    :return: Path to the saved file.
    """
    # validate file_format
    file_format = WeightsFileFormat(file_format)

    suffix = ".npz" if file_format == WeightsFileFormat.NUMPY else f".{file_format}"
    path = str(path)
    if not path.endswith(suffix):
        # do this this instead of using Path.with_suffix because it will remove periods in the path
        path += suffix
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == WeightsFileFormat.PT:
        import torch

        weights = {k: torch.from_numpy(v) for k, v in weights.items()}
        torch.save(weights, path)
    elif file_format == WeightsFileFormat.NUMPY:
        import numpy as np

        np.savez(path, **weights)
    elif file_format == WeightsFileFormat.SAFETENSORS:
        from safetensors.numpy import save_file

        save_file(weights, path)
    elif file_format == WeightsFileFormat.ONNX_ADAPTER:
        import onnxruntime as ort
        from packaging import version

        if version.parse(ort.__version__) < version.parse("1.20"):
            raise ValueError("Saving ONNX adapter files is only supported in ONNX Runtime >= 1.20")

        adapter_format = ort.AdapterFormat()
        # TODO(jambayk): Add model and adapter version
        adapter_format.set_parameters({k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in weights.items()})
        adapter_format.export_adapter(str(path))

    return path


def load_weights(path: Union[str, Path], file_format: Optional[WeightsFileFormat] = None, framework: str = "numpy"):
    """Load weights from a file.

    :param path: Path to the file.
    :param file_format: Format of the file. If None, will try to infer from the file extension.
    :param framework: Framework to load the weights into. Supported values are "pt" (pytorch) and "numpy".
    :return: Weights.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if file_format is not None:
        pass
    elif path.suffix.startswith(".pt"):
        file_format = "pt"
    elif path.suffix.startswith(".np"):
        file_format = "numpy"
    elif path.suffix == ".safetensors":
        file_format = "safetensors"
    elif path.suffix == ".onnx_adapter":
        file_format = "onnx_adapter"
    else:
        raise ValueError(f"Unknown file format for {path}. Please provide file_format.")

    # validate file_format
    file_format = WeightsFileFormat(file_format)

    weights = None
    if file_format == WeightsFileFormat.PT:
        import torch

        weights = torch.load(path, weights_only=True)
        if framework == "numpy":
            weights = {k: v.numpy() for k, v in weights.items()}
    elif file_format == WeightsFileFormat.NUMPY:
        import numpy as np

        weights = dict(np.load(path))
        if framework == "pt":
            import torch

            weights = {k: torch.from_numpy(v) for k, v in weights.items()}
    elif file_format == WeightsFileFormat.SAFETENSORS:
        from safetensors import safe_open

        weights = {}
        with safe_open(path, framework=framework, device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                weights[key] = f.get_tensor(key)
    elif file_format == WeightsFileFormat.ONNX_ADAPTER:
        import numpy as np
        import onnxruntime as ort
        from packaging import version

        if version.parse(ort.__version__) < version.parse("1.20"):
            raise ValueError("Loading ONNX adapter files is only supported in ONNX Runtime >= 1.20")

        adapter_format = ort.AdapterFormat.read_adapter(str(path))
        # need to do np.copy since the .numpy is bound to the ort.OrtValue
        # after returning the weights, adapter_format will be deleted so the numpy will be invalid
        weights = {k: np.copy(v.numpy()) for k, v in adapter_format.get_parameters().items()}
        if framework == "pt":
            import torch

            weights = {k: torch.from_numpy(v) for k, v in weights.items()}

    return weights


def unescaped_str(arg_str):
    """Decode strings without escaping."""
    return codecs.decode(arg_str, "unicode_escape")


def cleanup_memory():
    """Cleanup memory by running garbage collection and emptying CUDA cache."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
