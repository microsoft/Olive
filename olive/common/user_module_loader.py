# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any, Callable, Optional, Union

from olive.common.import_lib import import_user_module


class UserModuleLoader:
    """Load user module and call object in it.

    Only used for objects that are not json serializable.
    """

    def __init__(self, user_script: Optional[Union[Path, str]], script_dir: Optional[Union[Path, str]] = None):
        self.user_script = user_script
        self.script_dir = script_dir
        if self.user_script:
            self.user_module = import_user_module(user_script, script_dir)
            sys.modules[self.user_module.__name__] = self.user_module
        else:
            self.user_module = None

    def call_object(self, obj: Union[str, Callable, Any], *args, **kwargs):
        """Call obj with given arguments if it is a function, otherwise just return the object."""
        obj = self.load_object(obj)
        # We check for FunctionType, MethodType here instead of Callable since objects with __call__ methods
        # like torch.nn.module are also Callables
        if isinstance(obj, (FunctionType, MethodType)):
            return obj(*args, **kwargs)
        return obj

    def load_object(self, obj: Union[str, Callable, Any]):
        """Get obj from user_module if it is string name else return object."""
        if isinstance(obj, str):
            assert self.user_module is not None, "There is no user module to load object from."
            return getattr(self.user_module, obj)
        return obj

    def has_function(self, func_name: str) -> bool:
        return hasattr(self.user_module, func_name) and callable(getattr(self.user_module, func_name))
