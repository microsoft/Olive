# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import functools
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Union


@functools.lru_cache
def import_module_from_file(module_path: Union[Path, str], module_name: Optional[str] = None):
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise ValueError(f"{module_path} doesn't exist")

    if module_name is None:
        if module_path.is_dir():
            module_name = module_path.name
            module_path = module_path / "__init__.py"
        elif module_path.name == "__init__.py":
            module_name = module_path.parent.name
        else:
            module_name = module_path.stem

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    new_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_module)
    return new_module


@functools.lru_cache
def import_user_module(user_script: Union[Path, str], script_dir: Optional[Union[Path, str]] = None):
    if script_dir is not None:
        script_dir = Path(script_dir).resolve()
        if not script_dir.exists():
            raise ValueError(f"{script_dir} doesn't exist")
        if str(script_dir) not in sys.path:
            sys.path.append(str(script_dir))

    return import_module_from_file(user_script)
